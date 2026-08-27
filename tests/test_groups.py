from __future__ import annotations

import re

import polars as pl
import pytest

from ecup.data.groups import (
    GROUP_COLUMNS,
    build_group_artifacts,
    build_group_mapping,
    main,
    validate_group_mapping,
)


def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "": list(range(8)),
            "id": [10, 11, 12, 13, 14, 20, 21, 30],
            "name": [
                "Product A",
                "Product A",
                " product a ",
                "Product A",
                "Unique",
                "Exact",
                "Exact",
                "Other",
            ],
            "description": [
                "Same",
                "Same",
                " Same!! ",
                "Same",
                "Only",
                "Raw",
                "Raw",
                None,
            ],
            "category": [
                "БАД",
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
            ],
            "label": [1, 0, 1, 0, 0, 1, 1, 1],
        }
    )


def rows_by_id(mapping: pl.DataFrame) -> dict[int, dict[str, object]]:
    return {
        row["id"]: row
        for row in mapping.iter_rows(named=True)
    }


def test_mapping_assigns_exact_normalized_and_unique_groups() -> None:
    mapping = build_group_mapping(sample_frame())
    rows = rows_by_id(mapping)

    assert mapping.columns == list(GROUP_COLUMNS)
    assert mapping.height == sample_frame().height
    assert mapping.get_column("id").n_unique() == mapping.height

    normalized_group = {rows[item_id]["group_id"] for item_id in (10, 11, 12, 13)}
    assert len(normalized_group) == 1
    assert rows[10]["group_size"] == 4
    assert rows[10]["duplicate_kind"] == "normalized"
    assert re.fullmatch(r"g_[0-9a-f]{24}", rows[10]["group_id"])

    exact_group = {rows[item_id]["group_id"] for item_id in (20, 21)}
    assert len(exact_group) == 1
    assert rows[20]["group_size"] == 2
    assert rows[20]["duplicate_kind"] == "exact"

    assert rows[14]["duplicate_kind"] == "unique"
    assert rows[30]["duplicate_kind"] == "unique"
    assert rows[14]["group_id"] != rows[30]["group_id"]


def test_label_conflict_is_scoped_to_category() -> None:
    artifacts = build_group_artifacts(sample_frame())
    rows = rows_by_id(artifacts.mapping)

    assert rows[10]["label_conflict"] is True
    assert rows[11]["label_conflict"] is True
    assert rows[12]["label_conflict"] is True
    assert rows[13]["label_conflict"] is False
    assert artifacts.conflicts.get_column("id").to_list() == [10, 11, 12]
    assert artifacts.conflicts.columns == artifacts.mapping.columns


def test_group_ids_do_not_depend_on_input_order() -> None:
    forward = build_group_mapping(sample_frame()).select("id", "group_id")
    reversed_mapping = build_group_mapping(sample_frame().reverse()).select(
        "id", "group_id"
    )

    assert forward.to_dicts() == reversed_mapping.to_dicts()


def test_duplicate_source_ids_are_rejected() -> None:
    frame = sample_frame().with_columns(
        pl.when(pl.col("id") == 11)
        .then(pl.lit(10))
        .otherwise(pl.col("id"))
        .alias("id")
    )

    with pytest.raises(ValueError, match="id values must be unique"):
        build_group_mapping(frame)


def test_mapping_rejects_incorrect_recorded_group_size() -> None:
    source = sample_frame()
    mapping = build_group_mapping(source)
    target_group = mapping.filter(pl.col("id") == 10).get_column("group_id").item()
    broken = mapping.with_columns(
        pl.when(pl.col("group_id") == target_group)
        .then(pl.lit(999, dtype=pl.UInt32))
        .otherwise(pl.col("group_size"))
        .alias("group_size")
    )

    with pytest.raises(ValueError, match="actual group size"):
        validate_group_mapping(source, broken)

def test_cli_writes_parquet_contract_and_report(tmp_path) -> None:
    input_path = tmp_path / "data.csv"
    groups_path = tmp_path / "duplicate_groups.parquet"
    conflicts_path = tmp_path / "label_conflicts.parquet"
    report_path = tmp_path / "R04-groups.md"
    sample_frame().write_csv(input_path)

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--groups-output",
            str(groups_path),
            "--conflicts-output",
            str(conflicts_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    mapping = pl.read_parquet(groups_path)
    conflicts = pl.read_parquet(conflicts_path)
    assert mapping.columns == list(GROUP_COLUMNS)
    assert conflicts.columns == list(GROUP_COLUMNS)
    assert mapping.height == sample_frame().height
    assert conflicts.get_column("id").to_list() == [10, 11, 12]

    report = report_path.read_text(encoding="utf-8")
    assert "# R04 — группы дублей и `group_id`" in report
    assert "Групп точных дублей: **1**" in report
    assert "Групп нормализованных дублей: **1**" in report
