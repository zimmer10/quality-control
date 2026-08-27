from __future__ import annotations

import polars as pl
import pytest

from ecup.data.audit import (
    audit_dataset,
    exact_duplicate_groups,
    main,
    near_duplicate_groups,
    validation_summary,
)


def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "": [0, 1, 2, 3, 4],
            "id": [10, 11, 12, 13, 14],
            "name": ["Product A", "Product A", " product a ", "Grill", "Fuel"],
            "description": ["Same", "Same", " Same!! ", "Device", None],
            "category": [
                "БАД",
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
            ],
            "label": [1, 0, 1, 0, 1],
        }
    )


def test_audit_detects_balance_missing_and_exported_index() -> None:
    result = audit_dataset(sample_frame())

    assert result.validation.row_count == 5
    assert result.validation.unique_id_count == 5
    assert result.validation.unexpected_columns == ("",)
    assert result.validation.index_like_columns == ("",)
    assert result.class_balance.to_dicts() == [
        {"category": "БАД", "label": 0, "count": 1, "share": 1 / 3},
        {"category": "БАД", "label": 1, "count": 2, "share": 2 / 3},
        {
            "category": "Легковоспламеняющиеся",
            "label": 0,
            "count": 1,
            "share": 0.5,
        },
        {
            "category": "Легковоспламеняющиеся",
            "label": 1,
            "count": 1,
            "share": 0.5,
        },
    ]
    assert result.missing_values.to_dicts() == [
        {"column": "name", "null_count": 0, "blank_count": 0, "missing_total": 0},
        {
            "column": "description",
            "null_count": 1,
            "blank_count": 0,
            "missing_total": 1,
        },
    ]


def test_duplicate_groups_distinguish_exact_and_normalized_variants() -> None:
    exact = exact_duplicate_groups(sample_frame())
    near = near_duplicate_groups(sample_frame())

    assert exact.height == 1
    assert exact.row(0, named=True)["ids"] == [10, 11]
    assert exact.row(0, named=True)["labels"] == [0, 1]
    assert exact.row(0, named=True)["label_conflict"] is True

    assert near.height == 1
    assert near.row(0, named=True)["ids"] == [10, 11, 12]
    assert near.row(0, named=True)["raw_variants"] == 2
    assert near.row(0, named=True)["label_conflict"] is True


def test_missing_required_column_is_rejected() -> None:
    frame = sample_frame().drop("label")

    with pytest.raises(ValueError, match="missing required columns: label"):
        validation_summary(frame)


def test_cli_writes_markdown_report(tmp_path) -> None:
    input_path = tmp_path / "data.csv"
    output_path = tmp_path / "audit.md"
    sample_frame().write_csv(input_path)

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    report = output_path.read_text(encoding="utf-8")
    assert "# R03 — аудит `data.csv`" in report
    assert "`label = 1` — товар качественный" in report
    assert "Групп точных дублей: **1**" in report
