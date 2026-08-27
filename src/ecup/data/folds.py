"""Deterministic group-safe stratified fold construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import yaml

FOLD_COLUMNS = ("id", "category", "label", "group_id", "fold")
SOURCE_COLUMNS = ("id", "category", "label")
GROUP_COLUMNS = (*SOURCE_COLUMNS, "group_id")


@dataclass(frozen=True)
class FoldConfig:
    """Parameters controlling the shared fold assignment."""

    n_splits: int = 5
    seed: int = 42


def load_fold_config(path: str | Path) -> FoldConfig:
    """Read and validate fold parameters from the shared YAML config."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fold config must be a YAML mapping")
    try:
        n_splits = int(payload["n_splits"])
        seed = int(payload["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fold config must contain integer n_splits and seed") from error
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    return FoldConfig(n_splits=n_splits, seed=seed)


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} misses required columns: {', '.join(missing)}"
        )


def validate_fold_inputs(
    source: pl.DataFrame,
    groups: pl.DataFrame,
    n_splits: int,
) -> None:
    """Validate the R04 mapping before it becomes the shared R05 contract."""

    _require_columns(source, SOURCE_COLUMNS, "source")
    _require_columns(groups, GROUP_COLUMNS, "groups")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if source.is_empty():
        raise ValueError("source dataset is empty")
    if source.get_column("id").null_count() or groups.get_column("id").null_count():
        raise ValueError("id contains null values")
    if source.get_column("id").n_unique() != source.height:
        raise ValueError("source id values must be unique")
    if groups.get_column("id").n_unique() != groups.height:
        raise ValueError("group mapping id values must be unique")
    if source.height != groups.height:
        raise ValueError("source and group mapping row counts differ")
    if set(source.get_column("id").to_list()) != set(
        groups.get_column("id").to_list()
    ):
        raise ValueError("source and group mapping id sets differ")
    if groups.get_column("group_id").null_count():
        raise ValueError("group_id contains null values")
    if groups.schema["group_id"] != pl.String:
        raise ValueError("group_id must use the String dtype")
    if groups.get_column("group_id").n_unique() < n_splits:
        raise ValueError("number of groups is smaller than n_splits")
    for column in ("category", "label"):
        if source.get_column(column).null_count():
            raise ValueError(f"source {column} contains null values")
        if groups.get_column(column).null_count():
            raise ValueError(f"group mapping {column} contains null values")

    comparison = source.select(SOURCE_COLUMNS).join(
        groups.select(GROUP_COLUMNS),
        on="id",
        how="inner",
        suffix="_group",
    )
    mismatches = comparison.filter(
        (pl.col("category") != pl.col("category_group"))
        | (pl.col("label") != pl.col("label_group"))
    )
    if not mismatches.is_empty():
        raise ValueError("source and group mapping category/label values differ")


def _seeded_hash(seed: int, namespace: str, value: object) -> str:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_vectors(
    groups: pl.DataFrame,
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    strata = sorted(
        {
            (str(row["category"]), int(row["label"]))
            for row in groups.select("category", "label").iter_rows(named=True)
        }
    )
    stratum_index = {value: index for index, value in enumerate(strata)}
    counts = groups.group_by(["group_id", "category", "label"]).agg(
        pl.len().alias("count")
    )

    vectors: dict[str, np.ndarray] = {}
    for row in counts.iter_rows(named=True):
        group_id = str(row["group_id"])
        vector = vectors.setdefault(
            group_id,
            np.zeros(len(strata), dtype=np.int64),
        )
        vector[stratum_index[(str(row["category"]), int(row["label"]))]] = int(
            row["count"]
        )

    items = list(vectors.items())
    totals = np.sum(np.stack([vector for _, vector in items]), axis=0)
    if np.any(totals == 0):
        raise ValueError("every discovered stratum must contain at least one row")
    return items, totals


def assign_groups_to_folds(
    groups: pl.DataFrame,
    config: FoldConfig,
) -> dict[str, int]:
    """Greedily minimize stratification imbalance without splitting groups."""

    items, totals = _group_vectors(groups)
    items.sort(
        key=lambda item: (
            -float(np.std(item[1])),
            -int(item[1].sum()),
            _seeded_hash(config.seed, "group", item[0]),
        )
    )

    fold_counts = np.zeros(
        (config.n_splits, len(totals)),
        dtype=np.int64,
    )
    fold_sizes = np.zeros(config.n_splits, dtype=np.int64)
    fold_order = sorted(
        range(config.n_splits),
        key=lambda fold: _seeded_hash(config.seed, "fold", fold),
    )
    fold_rank = {fold: rank for rank, fold in enumerate(fold_order)}
    assignments: dict[str, int] = {}

    for group_id, vector in items:
        best_choice: tuple[tuple[float, int, int], int] | None = None
        for fold in range(config.n_splits):
            fold_counts[fold] += vector
            imbalance = float(
                np.mean(np.std(fold_counts / totals, axis=0))
            )
            fold_counts[fold] -= vector
            score = (
                imbalance,
                int(fold_sizes[fold]),
                fold_rank[fold],
            )
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, fold)

        if best_choice is None:
            raise RuntimeError("failed to select a fold")
        selected_fold = best_choice[1]
        fold_counts[selected_fold] += vector
        fold_sizes[selected_fold] += int(vector.sum())
        assignments[group_id] = selected_fold

    return assignments


def validate_folds(
    source: pl.DataFrame,
    groups: pl.DataFrame,
    folds: pl.DataFrame,
    config: FoldConfig,
) -> None:
    """Enforce the frozen folds.parquet schema and leak-free invariants."""

    if folds.columns != list(FOLD_COLUMNS):
        raise ValueError(f"fold columns must be exactly: {FOLD_COLUMNS}")
    if folds.height != source.height:
        raise ValueError("folds must contain exactly one row per source row")
    if folds.get_column("id").n_unique() != folds.height:
        raise ValueError("fold id values must be unique")
    if set(folds.get_column("id").to_list()) != set(
        source.get_column("id").to_list()
    ):
        raise ValueError("fold and source id sets differ")
    if folds.null_count().row(0) != (0, 0, 0, 0, 0):
        raise ValueError("fold contract contains null values")

    fold_values = set(folds.get_column("fold").to_list())
    expected_folds = set(range(config.n_splits))
    if fold_values != expected_folds:
        raise ValueError(
            f"fold values must be {sorted(expected_folds)}, got {sorted(fold_values)}"
        )

    leaking_groups = (
        folds.group_by("group_id")
        .agg(pl.col("fold").n_unique().alias("fold_count"))
        .filter(pl.col("fold_count") != 1)
    )
    if not leaking_groups.is_empty():
        raise ValueError("a group_id is assigned to multiple folds")

    expected = groups.select(GROUP_COLUMNS).sort("id")
    actual = folds.select(GROUP_COLUMNS).sort("id")
    if not expected.equals(actual):
        raise ValueError("fold rows do not match the R04 group mapping")


def build_folds(
    source: pl.DataFrame,
    groups: pl.DataFrame,
    config: FoldConfig,
) -> pl.DataFrame:
    """Create the canonical row-level folds table."""

    validate_fold_inputs(source, groups, config.n_splits)
    contract = groups.select(GROUP_COLUMNS).sort(["group_id", "id"])
    assignments = assign_groups_to_folds(contract, config)
    folds = (
        contract.with_columns(
            pl.col("group_id")
            .replace_strict(assignments, return_dtype=pl.Int8)
            .alias("fold")
        )
        .select(FOLD_COLUMNS)
        .sort("id")
    )
    validate_folds(source, groups, folds, config)
    return folds


def assignment_checksum(folds: pl.DataFrame) -> str:
    """Hash the canonical logical assignment independently of Parquet encoding."""

    digest = hashlib.sha256()
    for row in folds.select(FOLD_COLUMNS).sort("id").iter_rows():
        encoded = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _balance_rows(
    folds: pl.DataFrame,
    n_splits: int,
) -> tuple[list[str], float]:
    stats = folds.group_by(["category", "label", "fold"]).agg(
        pl.len().alias("count")
    )
    strata = sorted(
        {
            (str(row["category"]), int(row["label"]))
            for row in stats.select("category", "label").iter_rows(named=True)
        }
    )
    lines: list[str] = []
    maximum_deviation = 0.0
    for category, label in strata:
        subset = stats.filter(
            (pl.col("category") == category) & (pl.col("label") == label)
        )
        counts_by_fold = {
            int(row["fold"]): int(row["count"])
            for row in subset.iter_rows(named=True)
        }
        counts = [counts_by_fold.get(fold, 0) for fold in range(n_splits)]
        total = sum(counts)
        expected = total / n_splits
        deviation = max(abs(count - expected) / expected for count in counts)
        maximum_deviation = max(maximum_deviation, deviation)
        lines.append(
            "| "
            + " | ".join(
                [category, str(label), *(str(count) for count in counts), str(total)]
            )
            + " |"
        )
    return lines, maximum_deviation


def render_report(
    folds: pl.DataFrame,
    config: FoldConfig,
    input_path: str | Path,
    groups_path: str | Path,
    config_path: str | Path,
) -> str:
    """Render tracked split statistics and the canonical assignment checksum."""

    fold_sizes = {
        int(row["fold"]): int(row["count"])
        for row in (
            folds.group_by("fold")
            .agg(pl.len().alias("count"))
            .sort("fold")
            .iter_rows(named=True)
        )
    }
    balance_lines, maximum_deviation = _balance_rows(
        folds,
        config.n_splits,
    )
    group_sizes = folds.group_by("group_id").agg(pl.len().alias("size"))
    checksum = assignment_checksum(folds)
    fold_headers = [f"Fold {fold}" for fold in range(config.n_splits)]

    lines = [
        "# R05 — зафиксированное разбиение на folds",
        "",
        "## Метод",
        "",
        "- Единица разбиения — целый `group_id`, а не отдельная строка.",
        "- Балансируется сочетание `category + label`.",
        "- Группы назначаются жадно в fold с минимальным текущим дисбалансом.",
        f"- `seed = {config.seed}` задаёт детерминированный порядок равнозначных групп.",
        "- Входные строки предварительно приводятся к стабильному порядку.",
        "",
        "## Входные данные и конфигурация",
        "",
        f"- Датасет: `{input_path}`.",
        f"- Группы R04: `{groups_path}`.",
        f"- Конфигурация: `{config_path}`.",
        f"- Количество folds: **{config.n_splits}**.",
        f"- Строк: **{folds.height}**.",
        f"- Групп: **{folds.get_column('group_id').n_unique()}**.",
        f"- Максимальный размер группы: **{int(group_sizes.get_column('size').max())}**.",
        "- Групп, попавших в несколько folds: **0**.",
        "",
        "## Размер folds",
        "",
        "| Fold | Строк |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {fold} | {fold_sizes.get(fold, 0)} |"
        for fold in range(config.n_splits)
    )
    lines.extend(
        [
            "",
            "## Баланс `category × label`",
            "",
            "| Категория | Label | "
            + " | ".join(fold_headers)
            + " | Всего |",
            "|---|---:|" + "---:|" * (config.n_splits + 1),
        ]
    )
    lines.extend(balance_lines)
    lines.extend(
        [
            "",
            f"Максимальное относительное отклонение от идеального количества: **{maximum_deviation:.2%}**.",
            "",
            "## Схема `folds.parquet`",
            "",
            "| Столбец | Тип Polars |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| `{name}` | `{dtype}` |" for name, dtype in folds.schema.items()
    )
    lines.extend(
        [
            "",
            "## Контрольная сумма",
            "",
            "SHA-256 логического содержимого, отсортированного по `id`:",
            "",
            f"```text\n{checksum}\n```",
            "",
            "Контрольная сумма фиксирует само назначение folds и не зависит от особенностей Parquet-кодека.",
            "",
            "Файл `folds.parquet` остаётся локальным артефактом. Оба разработчика должны воспроизводить одинаковую контрольную сумму перед расчётом OOF.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic group-safe folds for E-CUP"
    )
    parser.add_argument("--input", type=Path, default="data/raw/data.csv")
    parser.add_argument(
        "--groups",
        type=Path,
        default="data/processed/duplicate_groups.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/folds.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default="reports/R05-folds.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = pl.read_csv(args.input, infer_schema_length=10_000)
    groups = pl.read_parquet(args.groups)
    config = load_fold_config(args.config)
    folds = build_folds(source, groups, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    folds.write_parquet(args.output, compression="zstd")
    args.report.write_text(
        render_report(
            folds,
            config,
            args.input,
            args.groups,
            args.config,
        ),
        encoding="utf-8",
    )

    print(
        f"Folds complete: {folds.height} rows, "
        f"{folds.get_column('group_id').n_unique()} groups, "
        f"checksum={assignment_checksum(folds)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

