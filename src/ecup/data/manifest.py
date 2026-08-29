"""Build the final product-level data contract after visual deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from ecup.data.audit import load_dataset, normalized_text
from ecup.data.folds import FOLD_COLUMNS, assignment_checksum
from ecup.data.groups import GROUP_COLUMNS
from ecup.data.visual_duplicates import visual_pair_checksum

MANIFEST_COLUMNS = (
    "id",
    "name",
    "description",
    "category",
    "label",
    "normalized_name",
    "normalized_description",
    "group_id",
    "duplicate_type",
    "label_conflict",
    "image_count",
    "fold",
)
MANIFEST_SCHEMA = {
    "id": pl.Int64,
    "name": pl.String,
    "description": pl.String,
    "category": pl.String,
    "label": pl.Int64,
    "normalized_name": pl.String,
    "normalized_description": pl.String,
    "group_id": pl.String,
    "duplicate_type": pl.String,
    "label_conflict": pl.Boolean,
    "image_count": pl.UInt32,
    "fold": pl.Int8,
}
SOURCE_COLUMNS = ("id", "name", "description", "category", "label")
IMAGE_COLUMNS = ("id", "in_dataset", "status")


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


def _validate_unique_ids(frame: pl.DataFrame, table_name: str) -> None:
    if frame.get_column("id").null_count():
        raise ValueError(f"{table_name} id contains null values")
    if frame.get_column("id").n_unique() != frame.height:
        raise ValueError(f"{table_name} id values must be unique")


def _validate_source_alignment(
    source: pl.DataFrame,
    other: pl.DataFrame,
    table_name: str,
) -> None:
    _validate_unique_ids(other, table_name)
    if source.height != other.height:
        raise ValueError(f"source and {table_name} row counts differ")
    comparison = source.select("id", "category", "label").join(
        other.select("id", "category", "label"),
        on="id",
        how="inner",
        suffix="_other",
    )
    if comparison.height != source.height:
        raise ValueError(f"source and {table_name} id sets differ")
    mismatches = comparison.filter(
        (pl.col("category") != pl.col("category_other"))
        | (pl.col("label") != pl.col("label_other"))
    )
    if not mismatches.is_empty():
        raise ValueError(
            f"source and {table_name} category/label values differ"
        )


def _readable_image_counts(
    image_manifest: pl.DataFrame,
    dataset_ids: set[int],
) -> pl.DataFrame:
    _require_columns(image_manifest, IMAGE_COLUMNS, "image manifest")
    invalid_membership = image_manifest.filter(
        pl.col("in_dataset")
        & (
            pl.col("id").is_null()
            | ~pl.col("id").is_in(sorted(dataset_ids))
        )
    )
    if not invalid_membership.is_empty():
        raise ValueError("image manifest marks unknown ids as in_dataset")
    return (
        image_manifest.filter(
            pl.col("in_dataset") & (pl.col("status") == "ok")
        )
        .group_by("id")
        .agg(pl.len().cast(pl.UInt32).alias("image_count"))
    )


def validate_product_manifest(
    source: pl.DataFrame,
    groups: pl.DataFrame,
    folds: pl.DataFrame,
    image_manifest: pl.DataFrame,
    manifest: pl.DataFrame,
) -> None:
    """Validate all joins and leak-safe invariants of the final manifest."""

    if manifest.columns != list(MANIFEST_COLUMNS):
        raise ValueError(
            f"manifest columns must be exactly: {MANIFEST_COLUMNS}"
        )
    if dict(manifest.schema) != MANIFEST_SCHEMA:
        raise ValueError("manifest dtypes do not match the frozen schema")
    if manifest.height != source.height:
        raise ValueError("manifest must contain exactly one row per source row")
    _validate_unique_ids(manifest, "manifest")
    if manifest.select(
        pl.any_horizontal(
            pl.col("category").is_null(),
            pl.col("label").is_null(),
            pl.col("normalized_name").is_null(),
            pl.col("normalized_description").is_null(),
            pl.col("group_id").is_null(),
            pl.col("duplicate_type").is_null(),
            pl.col("label_conflict").is_null(),
            pl.col("image_count").is_null(),
            pl.col("fold").is_null(),
        ).any()
    ).item():
        raise ValueError("manifest contains nulls in required derived fields")

    expected_text = (
        source.select(SOURCE_COLUMNS)
        .with_columns(
            normalized_text("name").alias("normalized_name"),
            normalized_text("description").alias("normalized_description"),
        )
        .sort("id")
    )
    if not expected_text.equals(
        manifest.select(
            *SOURCE_COLUMNS,
            "normalized_name",
            "normalized_description",
        ).sort("id")
    ):
        raise ValueError("manifest text fields do not match source")

    expected_groups = groups.select(
        "id",
        "group_id",
        pl.col("duplicate_kind").alias("duplicate_type"),
        "label_conflict",
    ).sort("id")
    if not expected_groups.equals(
        manifest.select(
            "id", "group_id", "duplicate_type", "label_conflict"
        ).sort("id")
    ):
        raise ValueError("manifest group fields do not match final groups")

    expected_folds = folds.select("id", "fold").sort("id")
    if not expected_folds.equals(manifest.select("id", "fold").sort("id")):
        raise ValueError("manifest fold values do not match folds")

    expected_counts = (
        source.select("id")
        .join(
            _readable_image_counts(
                image_manifest,
                set(source.get_column("id").to_list()),
            ),
            on="id",
            how="left",
        )
        .with_columns(
            pl.col("image_count").fill_null(0).cast(pl.UInt32)
        )
        .sort("id")
    )
    if not expected_counts.equals(
        manifest.select("id", "image_count").sort("id")
    ):
        raise ValueError("manifest image_count values are incorrect")

    leaks = (
        manifest.group_by("group_id")
        .agg(pl.col("fold").n_unique().alias("fold_count"))
        .filter(pl.col("fold_count") != 1)
    )
    if not leaks.is_empty():
        raise ValueError("a final group_id is assigned to multiple folds")


def build_product_manifest(
    source: pl.DataFrame,
    groups: pl.DataFrame,
    folds: pl.DataFrame,
    image_manifest: pl.DataFrame,
) -> pl.DataFrame:
    """Join source, final groups, final folds and readable image counts."""

    _require_columns(source, SOURCE_COLUMNS, "source")
    _require_columns(groups, GROUP_COLUMNS, "groups")
    _require_columns(folds, FOLD_COLUMNS, "folds")
    _validate_unique_ids(source, "source")
    _validate_source_alignment(source, groups, "groups")
    _validate_source_alignment(source, folds, "folds")

    group_folds = folds.group_by("group_id").agg(
        pl.col("fold").n_unique().alias("fold_count")
    )
    if group_folds.filter(pl.col("fold_count") != 1).height:
        raise ValueError("folds split at least one final group_id")

    counts = _readable_image_counts(
        image_manifest,
        set(source.get_column("id").to_list()),
    )
    manifest = (
        source.select(SOURCE_COLUMNS)
        .with_columns(
            normalized_text("name").alias("normalized_name"),
            normalized_text("description").alias("normalized_description"),
        )
        .join(
            groups.select(
                "id",
                "group_id",
                pl.col("duplicate_kind").alias("duplicate_type"),
                "label_conflict",
            ),
            on="id",
            how="left",
        )
        .join(folds.select("id", "fold"), on="id", how="left")
        .join(counts, on="id", how="left")
        .with_columns(
            pl.col("image_count").fill_null(0).cast(pl.UInt32),
            pl.col("fold").cast(pl.Int8),
        )
        .select(MANIFEST_COLUMNS)
        .sort("id")
    )
    validate_product_manifest(
        source,
        groups,
        folds,
        image_manifest,
        manifest,
    )
    return manifest


def product_manifest_checksum(manifest: pl.DataFrame) -> str:
    """Hash canonical logical rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in manifest.select(MANIFEST_COLUMNS).sort("id").iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        str(row[column]): int(row["count"])
        for row in (
            frame.group_by(column)
            .agg(pl.len().alias("count"))
            .sort(column)
            .iter_rows(named=True)
        )
    }


def render_report(
    manifest: pl.DataFrame,
    groups: pl.DataFrame,
    folds: pl.DataFrame,
    visual_pairs: pl.DataFrame,
    visual_stats: Mapping[str, object],
) -> str:
    """Render the tracked R07 audit and final-contract summary."""

    pair_kinds = _counts(visual_pairs, "match_kind")
    group_kinds = _counts(
        groups.unique(subset=["group_id"]),
        "duplicate_kind",
    )
    fold_sizes = _counts(folds, "fold")
    auto_merge_pairs = visual_pairs.filter(pl.col("auto_merge")).height
    merge_groups = groups.filter(
        pl.col("duplicate_kind").is_in(["visual", "mixed"])
    ).get_column("group_id").n_unique()
    leaks = (
        folds.group_by("group_id")
        .agg(pl.col("fold").n_unique().alias("count"))
        .filter(pl.col("count") != 1)
        .height
    )

    lines = [
        "# R07 — визуальные дубли и финальные data contracts",
        "",
        "## Метод",
        "",
        "- Для каждого читаемого изображения рассчитаны SHA-256, pHash и dHash.",
        "- Точные совпадения формируют надёжные связи, perceptual hashes — кандидатов для анализа.",
        "- Автообъединение консервативно: требуется точное основное либо минимум два точных изображения.",
        "- Подтверждённые связи транзитивно объединены с текстовыми группами R04.",
        "- После финализации групп `folds.parquet` пересобран через `StratifiedGroupKFold`.",
        "",
        "## Визуальные совпадения",
        "",
        f"- Обработано изображений: **{visual_stats.get('image_count', '—')}**.",
        f"- Ошибок хеширования: **{visual_stats.get('hash_errors', '—')}**.",
        f"- Кандидатных пар товаров: **{visual_pairs.height}**.",
        f"- Подтверждённых связей `auto_merge`: **{auto_merge_pairs}**.",
        f"- Финальных visual/mixed групп: **{merge_groups}**.",
        f"- Время хеширования: **{visual_stats.get('hashing_seconds', '—')} с**.",
        f"- Время сопоставления: **{visual_stats.get('matching_seconds', '—')} с**.",
        "",
        "| Вид совпадения | Пар |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{kind}` | {count} |" for kind, count in pair_kinds.items()
    )
    lines.extend(
        [
            "",
            "## Финальные группы",
            "",
            f"- Всего групп: **{groups.get_column('group_id').n_unique()}**.",
        ]
    )
    lines.extend(
        f"- `{kind}`: **{count}** групп."
        for kind, count in group_kinds.items()
    )
    lines.extend(
        [
            "",
            "## Финальные folds",
            "",
            f"- Групп, попавших в несколько folds: **{leaks}**.",
        ]
    )
    lines.extend(
        f"- Fold {fold}: **{count}** строк."
        for fold, count in fold_sizes.items()
    )
    lines.extend(
        [
            "",
            f"- SHA-256 назначения folds: `{assignment_checksum(folds)}`.",
            "",
            "## Итоговый `manifest.parquet`",
            "",
            f"- Строк: **{manifest.height}**.",
            f"- Товаров без читаемых изображений: **{manifest.filter(pl.col('image_count') == 0).height}**.",
            f"- SHA-256 логического содержимого: `{product_manifest_checksum(manifest)}`.",
            "",
            "| Столбец | Тип Polars |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| `{name}` | `{dtype}` |"
        for name, dtype in manifest.schema.items()
    )
    lines.extend(
        [
            "",
            "## Контрольные суммы R07",
            "",
            f"- Image hashes: `{visual_stats.get('hash_checksum', '—')}`.",
            f"- Visual pairs: `{visual_stats.get('pair_checksum', visual_pair_checksum(visual_pairs))}`.",
            "",
            "Parquet-файлы и JSON со статистикой остаются локальными; код и этот отчёт сохраняются в Git.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final E-CUP product manifest"
    )
    parser.add_argument("--input", type=Path, default="data/raw/data.csv")
    parser.add_argument(
        "--groups",
        type=Path,
        default="data/processed/duplicate_groups.parquet",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default="data/processed/folds.parquet",
    )
    parser.add_argument(
        "--images-manifest",
        type=Path,
        default="data/processed/image_manifest.parquet",
    )
    parser.add_argument(
        "--visual-pairs",
        type=Path,
        default="data/processed/visual_duplicates.parquet",
    )
    parser.add_argument(
        "--visual-stats",
        type=Path,
        default="data/processed/visual_duplicate_stats.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/manifest.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default="reports/R07-visual-duplicates.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_dataset(args.input)
    groups = pl.read_parquet(args.groups)
    folds = pl.read_parquet(args.folds)
    image_manifest = pl.read_parquet(args.images_manifest)
    visual_pairs = pl.read_parquet(args.visual_pairs)
    visual_stats = json.loads(args.visual_stats.read_text(encoding="utf-8"))

    manifest = build_product_manifest(
        source,
        groups,
        folds,
        image_manifest,
    )
    report = render_report(
        manifest,
        groups,
        folds,
        visual_pairs,
        visual_stats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(args.output, compression="zstd")
    args.report.write_text(report, encoding="utf-8")

    print(
        f"Product manifest complete: {manifest.height} rows, "
        f"checksum={product_manifest_checksum(manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
