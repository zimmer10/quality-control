"""Deterministic duplicate groups used for leak-free dataset splitting."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from ecup.data.audit import (
    load_dataset,
    normalized_text,
    validation_summary,
)

GROUP_COLUMNS = (
    "id",
    "category",
    "label",
    "group_id",
    "group_size",
    "duplicate_kind",
    "label_conflict",
)


@dataclass(frozen=True)
class GroupArtifacts:
    """Row-level group mapping and the subset requiring label review."""

    mapping: pl.DataFrame
    conflicts: pl.DataFrame


def stable_group_id(fingerprint: str) -> str:
    """Return a stable identifier independent of input row order."""

    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return f"g_{digest}"


def _validate_source(frame: pl.DataFrame) -> None:
    summary = validation_summary(frame)
    problems: list[str] = []
    if frame.is_empty():
        problems.append("dataset is empty")
    if frame.get_column("id").null_count():
        problems.append("id contains null values")
    if summary.duplicate_id_rows:
        problems.append("id values must be unique")
    if frame.get_column("category").null_count():
        problems.append("category contains null values")
    if frame.get_column("label").null_count():
        problems.append("label contains null values")
    if summary.unknown_categories:
        problems.append(f"unknown categories: {summary.unknown_categories}")
    if summary.invalid_labels:
        problems.append(f"invalid labels: {summary.invalid_labels}")
    if problems:
        raise ValueError("; ".join(problems))


def _prepare_fingerprints(frame: pl.DataFrame) -> pl.DataFrame:
    prepared = frame.select(
        "id", "name", "description", "category", "label"
    ).with_columns(
        normalized_text("name").alias("_normalized_name"),
        normalized_text("description").alias("_normalized_description"),
        pl.concat_str(
            [
                pl.col("name").fill_null("<NULL>"),
                pl.col("description").fill_null("<NULL>"),
            ],
            separator="\u241f",
        ).alias("_raw_signature"),
    )
    prepared = prepared.with_columns(
        pl.concat_str(
            ["_normalized_name", "_normalized_description"],
            separator="\u241f",
        ).alias("_fingerprint")
    )

    fingerprints = prepared.get_column("_fingerprint").unique().to_list()
    lookup = pl.DataFrame(
        {
            "_fingerprint": fingerprints,
            "group_id": [stable_group_id(value) for value in fingerprints],
        },
        schema={"_fingerprint": pl.String, "group_id": pl.String},
    )
    if lookup.get_column("group_id").n_unique() != lookup.height:
        raise RuntimeError("group_id hash collision detected")
    return prepared.join(lookup, on="_fingerprint", how="left")


def validate_group_mapping(source: pl.DataFrame, mapping: pl.DataFrame) -> None:
    """Check the row-level contract consumed by R05."""

    missing_columns = sorted(set(GROUP_COLUMNS) - set(mapping.columns))
    if missing_columns:
        raise ValueError(f"mapping misses columns: {', '.join(missing_columns)}")
    if mapping.height != source.height:
        raise ValueError("mapping must contain exactly one row per source row")
    if mapping.get_column("id").n_unique() != mapping.height:
        raise ValueError("mapping id values must be unique")
    if set(mapping.get_column("id").to_list()) != set(
        source.get_column("id").to_list()
    ):
        raise ValueError("mapping and source id sets differ")
    if mapping.get_column("group_id").null_count():
        raise ValueError("group_id contains null values")
    size_checks = (
        mapping.group_by("group_id")
        .agg(
            pl.len().alias("actual_size"),
            pl.col("group_size").first().alias("recorded_size"),
            pl.col("group_size").n_unique().alias("size_variants"),
        )
    )
    if size_checks.filter(pl.col("size_variants") != 1).height:
        raise ValueError("group_size is inconsistent inside a group")
    if size_checks.filter(pl.col("actual_size") != pl.col("recorded_size")).height:
        raise ValueError("group_size does not match the actual group size")


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _final_group_lookup(
    mapping: pl.DataFrame,
    visual_pairs: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    required = {"left_id", "right_id", "auto_merge"}
    missing = sorted(required - set(visual_pairs.columns))
    if missing:
        raise ValueError(
            "visual pairs miss required columns: " + ", ".join(missing)
        )
    id_to_group = {
        int(row["id"]): str(row["group_id"])
        for row in mapping.select("id", "group_id").iter_rows(named=True)
    }
    pair_ids = set(visual_pairs.get_column("left_id").to_list()) | set(
        visual_pairs.get_column("right_id").to_list()
    )
    unknown_ids = sorted(pair_ids - set(id_to_group))
    if unknown_ids:
        raise ValueError(f"visual pairs contain unknown ids: {unknown_ids[:10]}")

    source_groups = sorted(set(id_to_group.values()))
    disjoint = _DisjointSet(source_groups)
    for row in visual_pairs.filter(pl.col("auto_merge")).iter_rows(named=True):
        disjoint.union(
            id_to_group[int(row["left_id"])],
            id_to_group[int(row["right_id"])],
        )

    components: dict[str, list[str]] = {}
    for group_id in source_groups:
        components.setdefault(disjoint.find(group_id), []).append(group_id)

    source_kinds = {
        str(row["group_id"]): str(row["duplicate_kind"])
        for row in (
            mapping.select("group_id", "duplicate_kind")
            .unique()
            .iter_rows(named=True)
        )
    }
    final_ids: dict[str, str] = {}
    final_kinds: dict[str, str] = {}
    for component in components.values():
        component.sort()
        if len(component) == 1:
            final_id = component[0]
            final_kind = source_kinds[component[0]]
        else:
            fingerprint = "visual\u241f" + "\u241f".join(component)
            final_id = stable_group_id(fingerprint)
            final_kind = (
                "visual"
                if all(source_kinds[value] == "unique" for value in component)
                else "mixed"
            )
        for source_group in component:
            final_ids[source_group] = final_id
        final_kinds[final_id] = final_kind

    if len(set(final_ids.values())) != len(components):
        raise RuntimeError("final group_id hash collision detected")
    return final_ids, final_kinds


def merge_visual_groups(
    mapping: pl.DataFrame,
    visual_pairs: pl.DataFrame,
) -> pl.DataFrame:
    """Merge confirmed visual edges into deterministic text groups."""

    missing = sorted(set(GROUP_COLUMNS) - set(mapping.columns))
    if missing:
        raise ValueError(f"mapping misses columns: {', '.join(missing)}")
    final_ids, final_kinds = _final_group_lookup(mapping, visual_pairs)
    merged = (
        mapping.drop("group_size", "duplicate_kind", "label_conflict")
        .with_columns(
            pl.col("group_id")
            .replace_strict(final_ids, return_dtype=pl.String)
            .alias("group_id")
        )
    )
    group_stats = merged.group_by("group_id").agg(
        pl.len().alias("group_size")
    ).with_columns(
        pl.col("group_id")
        .replace_strict(final_kinds, return_dtype=pl.String)
        .alias("duplicate_kind")
    )
    conflict_keys = (
        merged.group_by(["group_id", "category"])
        .agg(pl.col("label").n_unique().alias("_label_count"))
        .filter(pl.col("_label_count") > 1)
        .select(
            "group_id",
            "category",
            pl.lit(True).alias("label_conflict"),
        )
    )
    return (
        merged.join(group_stats, on="group_id", how="left")
        .join(conflict_keys, on=["group_id", "category"], how="left")
        .with_columns(pl.col("label_conflict").fill_null(False))
        .select(GROUP_COLUMNS)
        .sort("id")
    )

def build_group_mapping(frame: pl.DataFrame) -> pl.DataFrame:
    """Assign every item to an exact, normalized or unique text group."""

    _validate_source(frame)
    prepared = _prepare_fingerprints(frame)
    group_stats = (
        prepared.group_by("group_id")
        .agg(
            pl.len().alias("group_size"),
            pl.col("_raw_signature").n_unique().alias("_raw_variants"),
        )
        .with_columns(
            pl.when(pl.col("group_size") == 1)
            .then(pl.lit("unique"))
            .when(pl.col("_raw_variants") == 1)
            .then(pl.lit("exact"))
            .otherwise(pl.lit("normalized"))
            .alias("duplicate_kind")
        )
    )
    conflict_keys = (
        prepared.group_by(["group_id", "category"])
        .agg(pl.col("label").n_unique().alias("_label_count"))
        .filter(pl.col("_label_count") > 1)
        .select(
            "group_id",
            "category",
            pl.lit(True).alias("label_conflict"),
        )
    )

    mapping = (
        prepared.join(group_stats, on="group_id", how="left")
        .join(conflict_keys, on=["group_id", "category"], how="left")
        .with_columns(pl.col("label_conflict").fill_null(False))
        .select(GROUP_COLUMNS)
        .sort("id")
    )
    validate_group_mapping(frame, mapping)
    return mapping


def build_group_artifacts(
    frame: pl.DataFrame,
    visual_pairs: pl.DataFrame | None = None,
) -> GroupArtifacts:
    """Build the full mapping and a schema-identical conflict subset."""

    mapping = build_group_mapping(frame)
    if visual_pairs is not None:
        mapping = merge_visual_groups(mapping, visual_pairs)
        validate_group_mapping(frame, mapping)
    conflicts = mapping.filter(pl.col("label_conflict")).sort(
        ["group_id", "category", "id"]
    )
    return GroupArtifacts(mapping=mapping, conflicts=conflicts)


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _conflict_examples(
    source: pl.DataFrame,
    conflicts: pl.DataFrame,
    limit: int = 12,
) -> list[str]:
    examples = (
        conflicts.join(source.select("id", "name"), on="id", how="left")
        .select("group_id", "category", "id", "label", "name")
        .head(limit)
    )
    if examples.is_empty():
        return ["Конфликтов не найдено."]

    lines = [
        "| group_id | Категория | ID | Label | Название |",
        "|---|---|---:|---:|---|",
    ]
    for row in examples.iter_rows(named=True):
        name = row["name"] or ""
        if len(name) > 80:
            name = f"{name[:77]}..."
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(row[column])
                for column in ("group_id", "category", "id", "label")
            )
            + f" | {_markdown_escape(name)} |"
        )
    return lines


def render_report(
    source: pl.DataFrame,
    artifacts: GroupArtifacts,
    input_path: str | Path,
) -> str:
    """Render the tracked R04 summary; Parquet files remain local."""

    mapping = artifacts.mapping
    conflict_rows = artifacts.conflicts
    group_stats = mapping.unique(subset=["group_id"]).select(
        "group_id", "group_size", "duplicate_kind"
    )
    kind_counts = {
        row["duplicate_kind"]: row["count"]
        for row in (
            group_stats.group_by("duplicate_kind")
            .agg(pl.len().alias("count"))
            .iter_rows(named=True)
        )
    }
    duplicate_groups = group_stats.filter(pl.col("group_size") > 1)
    conflict_group_count = conflict_rows.select(
        "group_id", "category"
    ).unique().height

    lines = [
        "# R04 — группы дублей и `group_id`",
        "",
        "## Метод",
        "",
        "- `name` и `description` приводятся к нижнему регистру.",
        "- Пунктуация удаляется, последовательности пробелов нормализуются.",
        "- SHA-256 от нормализованной пары образует стабильный `group_id`.",
        "- Категория не входит в fingerprint: один товар должен оставаться в одном fold.",
        "- Конфликт `label` проверяется отдельно внутри каждой категории.",
        "- Метки автоматически не исправляются.",
        "",
        "Типы групп:",
        "",
        "- `unique` — в группе одна строка.",
        "- `exact` — несколько строк с полностью одинаковым исходным текстом.",
        "- `normalized` — исходные тексты различаются, но их нормализованные версии совпали.",
        "- `visual` — объединены ранее уникальные карточки с подтверждённым совпадением изображений.",
        "- `mixed` — визуальная связь объединила хотя бы одну группу текстовых дублей.",
        "",
        "На текстовом этапе группа с несколькими вариантами исходного текста считается `normalized`, даже если внутри неё есть точные повторы.",
        "После визуального объединения итоговый тип такой группы меняется на `mixed`.",
        "",
        "## Результат",
        "",
        f"- Входной файл: `{input_path}`.",
        f"- Строк во входе и mapping: **{mapping.height}**.",
        f"- Всего групп: **{group_stats.height}**.",
        f"- Уникальных групп: **{kind_counts.get('unique', 0)}**.",
        f"- Групп точных дублей: **{kind_counts.get('exact', 0)}**.",
        f"- Групп нормализованных дублей: **{kind_counts.get('normalized', 0)}**.",
        f"- Групп визуальных дублей: **{kind_counts.get('visual', 0)}**.",
        f"- Смешанных групп: **{kind_counts.get('mixed', 0)}**.",
        f"- Всего групп с несколькими строками: **{duplicate_groups.height}**.",
        f"- Строк в группах дублей: **{mapping.filter(pl.col('group_size') > 1).height}**.",
        f"- Максимальный размер группы: **{int(mapping.get_column('group_size').max())}**.",
        f"- Конфликтующих пар `group_id + category`: **{conflict_group_count}**.",
        f"- Строк в конфликтующих группах: **{conflict_rows.height}**.",
        "",
        "## Схема `duplicate_groups.parquet`",
        "",
        "| Столбец | Тип Polars |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{name}` | `{dtype}` |" for name, dtype in mapping.schema.items()
    )
    lines.extend(
        [
            "",
            "`label_conflicts.parquet` использует ту же схему и содержит только строки, где `label_conflict = true`.",
            "",
            "## Примеры конфликтующих строк",
            "",
        ]
    )
    lines.extend(_conflict_examples(source, conflict_rows))
    lines.extend(
        [
            "",
            "## Ограничения и следующий этап",
            "",
            "- Текстовый этап объединяет точные и консервативные нормализованные совпадения.",
            "- Если передан `visual_duplicates.parquet`, подтверждённые связи `auto_merge = true` объединяют текстовые группы транзитивно.",
            "- Семантически похожие товары с существенно различающимся текстом здесь не объединяются.",
            "- Неуверенные визуальные кандидаты сохраняются для анализа, но не влияют на `group_id`.",
            "- R05 должен использовать `group_id` как неделимую единицу при построении folds.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic duplicate groups for E-CUP"
    )
    parser.add_argument("--input", type=Path, default="data/raw/data.csv")
    parser.add_argument(
        "--groups-output",
        type=Path,
        default="data/processed/duplicate_groups.parquet",
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default="data/processed/label_conflicts.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default="reports/R04-groups.md",
    )
    parser.add_argument(
        "--visual-pairs",
        type=Path,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_dataset(args.input)
    visual_pairs = (
        pl.read_parquet(args.visual_pairs)
        if args.visual_pairs is not None
        else None
    )
    artifacts = build_group_artifacts(source, visual_pairs)

    args.groups_output.parent.mkdir(parents=True, exist_ok=True)
    args.conflicts_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mapping.write_parquet(args.groups_output, compression="zstd")
    artifacts.conflicts.write_parquet(args.conflicts_output, compression="zstd")
    args.report.write_text(
        render_report(source, artifacts, args.input),
        encoding="utf-8",
    )

    print(
        f"Groups complete: {artifacts.mapping.height} rows, "
        f"{artifacts.mapping.get_column('group_id').n_unique()} groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
