from __future__ import annotations

import json

import polars as pl
import pytest

from ecup.data.manifest import (
    MANIFEST_COLUMNS,
    build_product_manifest,
    main,
    product_manifest_checksum,
)
from ecup.data.visual_duplicates import PAIR_SCHEMA


def sample_inputs() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    source = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": [" Product A! ", "Product A", "Match", "Solo"],
            "description": ["TEXT", None, "Near", "Other"],
            "category": [
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
            ],
            "label": [0, 1, 0, 1],
        }
    )
    groups = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": source.get_column("category"),
            "label": source.get_column("label"),
            "group_id": ["g1", "g1", "g2", "g3"],
            "group_size": pl.Series([2, 2, 1, 1], dtype=pl.UInt32),
            "duplicate_kind": ["mixed", "mixed", "unique", "unique"],
            "label_conflict": [True, True, False, False],
        }
    )
    folds = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": source.get_column("category"),
            "label": source.get_column("label"),
            "group_id": ["g1", "g1", "g2", "g3"],
            "fold": pl.Series([0, 0, 1, 0], dtype=pl.Int8),
        }
    )
    image_manifest = pl.DataFrame(
        {
            "id": [1, 1, 2, 4],
            "in_dataset": [True, True, True, True],
            "status": ["ok", "ok", "error", "ok"],
        },
        schema={
            "id": pl.Int64,
            "in_dataset": pl.Boolean,
            "status": pl.String,
        },
    )
    return source, groups, folds, image_manifest


def sample_visual_pairs() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "left_id": 1,
                "right_id": 2,
                "left_path": "1/0.jpg",
                "right_path": "2/0.jpg",
                "match_kind": "exact",
                "exact_image_pairs": 1,
                "strong_image_pairs": 0,
                "candidate_image_pairs": 1,
                "phash_distance": 0,
                "dhash_distance": 0,
                "aspect_ratio_delta": 0.0,
                "primary_match": True,
                "auto_merge": True,
            }
        ],
        schema=PAIR_SCHEMA,
    )


def test_build_product_manifest_matches_pipeline_contract() -> None:
    source, groups, folds, image_manifest = sample_inputs()

    manifest = build_product_manifest(
        source,
        groups,
        folds,
        image_manifest,
    )

    assert manifest.columns == list(MANIFEST_COLUMNS)
    assert manifest.get_column("normalized_name").to_list() == [
        "product a",
        "product a",
        "match",
        "solo",
    ]
    assert manifest.get_column("normalized_description").to_list() == [
        "text",
        "",
        "near",
        "other",
    ]
    assert manifest.get_column("image_count").to_list() == [2, 0, 0, 1]
    assert manifest.get_column("duplicate_type").to_list() == [
        "mixed",
        "mixed",
        "unique",
        "unique",
    ]
    assert manifest.schema["image_count"] == pl.UInt32
    assert manifest.schema["fold"] == pl.Int8


def test_manifest_is_independent_of_input_order() -> None:
    source, groups, folds, image_manifest = sample_inputs()

    forward = build_product_manifest(
        source,
        groups,
        folds,
        image_manifest,
    )
    reversed_rows = build_product_manifest(
        source.reverse(),
        groups.reverse(),
        folds.reverse(),
        image_manifest.reverse(),
    )

    assert forward.equals(reversed_rows)
    assert product_manifest_checksum(forward) == product_manifest_checksum(
        reversed_rows
    )


def test_manifest_rejects_source_group_mismatch() -> None:
    source, groups, folds, image_manifest = sample_inputs()
    broken = groups.with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit(1))
        .otherwise(pl.col("label"))
        .alias("label")
    )

    with pytest.raises(ValueError, match="category/label values differ"):
        build_product_manifest(source, broken, folds, image_manifest)


def test_manifest_rejects_group_leak_between_folds() -> None:
    source, groups, folds, image_manifest = sample_inputs()
    broken = folds.with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.col("fold"))
        .alias("fold")
    )

    with pytest.raises(ValueError, match="split at least one"):
        build_product_manifest(source, groups, broken, image_manifest)


def test_manifest_rejects_unknown_in_dataset_image_id() -> None:
    source, groups, folds, image_manifest = sample_inputs()
    broken = image_manifest.vstack(
        pl.DataFrame(
            {"id": [999], "in_dataset": [True], "status": ["ok"]},
            schema=image_manifest.schema,
        )
    )

    with pytest.raises(ValueError, match="unknown ids"):
        build_product_manifest(source, groups, folds, broken)


def test_cli_writes_manifest_and_r07_report(tmp_path) -> None:
    source, groups, folds, image_manifest = sample_inputs()
    input_path = tmp_path / "data.csv"
    groups_path = tmp_path / "duplicate_groups.parquet"
    folds_path = tmp_path / "folds.parquet"
    images_path = tmp_path / "image_manifest.parquet"
    pairs_path = tmp_path / "visual_duplicates.parquet"
    stats_path = tmp_path / "visual_duplicate_stats.json"
    output_path = tmp_path / "manifest.parquet"
    report_path = tmp_path / "R07-visual-duplicates.md"

    source.write_csv(input_path)
    groups.write_parquet(groups_path)
    folds.write_parquet(folds_path)
    image_manifest.write_parquet(images_path)
    sample_visual_pairs().write_parquet(pairs_path)
    stats_path.write_text(
        json.dumps(
            {
                "image_count": 4,
                "hash_errors": 0,
                "hashing_seconds": 1.0,
                "matching_seconds": 0.1,
                "hash_checksum": "hashes",
                "pair_checksum": "pairs",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input", str(input_path),
            "--groups", str(groups_path),
            "--folds", str(folds_path),
            "--images-manifest", str(images_path),
            "--visual-pairs", str(pairs_path),
            "--visual-stats", str(stats_path),
            "--output", str(output_path),
            "--report", str(report_path),
        ]
    )

    assert exit_code == 0
    manifest = pl.read_parquet(output_path)
    assert manifest.columns == list(MANIFEST_COLUMNS)
    report = report_path.read_text(encoding="utf-8")
    assert "# R07 — визуальные дубли" in report
    assert "Групп, попавших в несколько folds: **0**" in report
    assert product_manifest_checksum(manifest) in report
