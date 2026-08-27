from __future__ import annotations

import polars as pl
import pytest

from ecup.data.folds import (
    FOLD_COLUMNS,
    FoldConfig,
    assignment_checksum,
    build_folds,
    load_fold_config,
    main,
    validate_fold_inputs,
)


def sample_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, object]] = []
    item_id = 0
    for category, label in (
        ("БАД", 0),
        ("БАД", 1),
        ("Легковоспламеняющиеся", 0),
        ("Легковоспламеняющиеся", 1),
    ):
        for group_number in range(3):
            rows.append(
                {
                    "id": item_id,
                    "name": f"item-{item_id}",
                    "description": "text",
                    "category": category,
                    "label": label,
                    "group_id": f"group-{category}-{label}-{group_number}",
                }
            )
            item_id += 1

    rows.extend(
        [
            {
                "id": item_id,
                "name": "conflict-a",
                "description": "same",
                "category": "БАД",
                "label": 0,
                "group_id": "conflicting-group",
            },
            {
                "id": item_id + 1,
                "name": "conflict-b",
                "description": "same",
                "category": "БАД",
                "label": 1,
                "group_id": "conflicting-group",
            },
        ]
    )
    full = pl.DataFrame(rows)
    source = full.select("id", "name", "description", "category", "label")
    groups = full.select("id", "category", "label", "group_id")
    return source, groups


def test_build_folds_preserves_groups_and_contract() -> None:
    source, groups = sample_inputs()
    config = FoldConfig(n_splits=3, seed=42)

    folds = build_folds(source, groups, config)

    assert folds.columns == list(FOLD_COLUMNS)
    assert folds.height == source.height
    assert folds.get_column("id").n_unique() == source.height
    assert set(folds.get_column("fold").to_list()) == {0, 1, 2}
    assert folds.schema["fold"] == pl.Int8
    leaks = (
        folds.group_by("group_id")
        .agg(pl.col("fold").n_unique().alias("fold_count"))
        .filter(pl.col("fold_count") != 1)
    )
    assert leaks.is_empty()
    conflict_folds = (
        folds.filter(pl.col("group_id") == "conflicting-group")
        .get_column("fold")
        .unique()
        .to_list()
    )
    assert len(conflict_folds) == 1


def test_assignment_is_independent_of_row_order() -> None:
    source, groups = sample_inputs()
    config = FoldConfig(n_splits=3, seed=42)

    forward = build_folds(source, groups, config)
    reversed_rows = build_folds(source.reverse(), groups.reverse(), config)

    assert forward.equals(reversed_rows)
    assert assignment_checksum(forward) == assignment_checksum(reversed_rows)


def test_seed_controls_deterministic_tie_breaking() -> None:
    source, groups = sample_inputs()

    seed_42 = build_folds(source, groups, FoldConfig(n_splits=3, seed=42))
    seed_43 = build_folds(source, groups, FoldConfig(n_splits=3, seed=43))

    assert not seed_42.equals(seed_43)
    assert assignment_checksum(seed_42) != assignment_checksum(seed_43)


def test_fold_input_mismatch_is_rejected() -> None:
    source, groups = sample_inputs()
    broken_groups = groups.with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit(1))
        .otherwise(pl.col("label"))
        .alias("label")
    )

    with pytest.raises(
        ValueError,
        match="category/label values differ",
    ):
        validate_fold_inputs(source, broken_groups, n_splits=3)


def test_fold_config_is_loaded_and_validated(tmp_path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text("seed: 17\nn_splits: 3\n", encoding="utf-8")

    assert load_fold_config(config_path) == FoldConfig(n_splits=3, seed=17)

    config_path.write_text("seed: 17\nn_splits: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2"):
        load_fold_config(config_path)


def test_cli_writes_folds_and_report(tmp_path) -> None:
    source, groups = sample_inputs()
    input_path = tmp_path / "data.csv"
    groups_path = tmp_path / "duplicate_groups.parquet"
    config_path = tmp_path / "data.yaml"
    output_path = tmp_path / "folds.parquet"
    report_path = tmp_path / "R05-folds.md"
    source.write_csv(input_path)
    groups.write_parquet(groups_path)
    config_path.write_text("seed: 42\nn_splits: 3\n", encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--groups",
            str(groups_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    folds = pl.read_parquet(output_path)
    assert folds.columns == list(FOLD_COLUMNS)
    assert folds.height == source.height
    assert folds.null_count().row(0) == (0, 0, 0, 0, 0)

    report = report_path.read_text(encoding="utf-8")
    assert "# R05 — зафиксированное разбиение на folds" in report
    assert assignment_checksum(folds) in report
    assert "Групп, попавших в несколько folds: **0**" in report
