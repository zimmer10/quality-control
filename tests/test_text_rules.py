from __future__ import annotations

from copy import deepcopy
import json

import polars as pl
import pytest
import yaml

from ecup.features.text import (
    TEXT_RULE_COLUMNS,
    TEXT_RULE_SCHEMA,
    apply_text_rules,
    build_text_rule_features,
    main,
    validate_text_rule_inputs,
)
from ecup.rules.engine import (
    EXPECTED_SIGNALS,
    load_text_rule_config,
    normalize_rule_text,
)
from ecup.rules.schema import load_evidence_schema


@pytest.fixture
def schema():
    return load_evidence_schema("configs/evidence.yaml")


@pytest.fixture
def config():
    return load_text_rule_config("configs/text_rules.yaml")


def apply(category: str, name: str, description: str | None, schema, config):
    return apply_text_rules(name, description, category, schema, config)


def sample_source() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": [
                "Витаминный комплекс",
                "BCAA",
                "Спички хозяйственные",
                "Газовая горелка",
            ],
            "description": [
                "Биологически активная добавка к пище",
                "Спортивное питание, аминокислоты",
                "Для разведения открытого огня",
                "Работает от баллона. Баллон приобретается отдельно",
            ],
            "category": [
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
            ],
        }
    )


def sample_folds(labels: list[int] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": [
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
            ],
            "label": labels or [1, 0, 1, 0],
            "group_id": ["g1", "g2", "g3", "g4"],
            "fold": pl.Series([0, 1, 0, 1], dtype=pl.Int8),
        }
    )


def test_canonical_text_rule_config_is_versioned_and_complete(config) -> None:
    assert config.version == 1
    assert config.exclusion_window == 80
    assert tuple(config.categories) == ("БАД", "Легковоспламеняющиеся")
    for category, expected_signals in EXPECTED_SIGNALS.items():
        assert tuple(config.category(category).signals) == expected_signals
        assert all(
            signal.patterns
            for signal in config.category(category).signals.values()
        )


def test_normalization_handles_html_entities_case_and_missing_text() -> None:
    assert normalize_rule_text("  БАД&nbsp; ДЛЯ&nbsp;ЕДЫ  ") == "бад для еды"
    assert normalize_rule_text("Ёлка") == "елка"
    assert normalize_rule_text(None) == ""


def test_bad_marker_builds_schema_valid_evidence(schema, config) -> None:
    result = apply(
        "БАД",
        "Витамины",
        "Продукт является биологически активной добавкой к пище",
        schema,
        config,
    )

    assert result.rule_id == "EXPLICIT_BAD_MARKER"
    assert result.verdict == 1
    assert result.values["explicit_bad_marker"] is True
    assert result.values["marker_source"] == "description"
    assert schema.build_target(
        result.category,
        result.values,
        result.rule_id,
        result.verdict,
    )["verdict"] == 1
    assert {record.source for record in result.evidence} == {"description"}


def test_explicit_bad_negation_suppresses_positive_marker(schema, config) -> None:
    result = apply(
        "БАД",
        "Травяной чай",
        "Не является биологически активной добавкой",
        schema,
        config,
    )

    assert result.rule_id == "EXPLICIT_NOT_BAD"
    assert result.verdict == 0
    assert result.values["explicit_not_bad"] is True
    assert result.values["explicit_bad_marker"] == schema.unknown_value
    assert result.values["marker_source"] == schema.unknown_value


def test_sport_nutrition_is_a_separate_negative_signal(schema, config) -> None:
    result = apply(
        "БАД",
        "BCAA 2:1:1",
        "Аминокислоты, спортивное питание",
        schema,
        config,
    )

    assert result.rule_id == "SPORT_NUTRITION"
    assert result.verdict == 0
    assert result.values["sport_nutrition"] is True
    assert result.values["explicit_bad_marker"] == schema.unknown_value


def test_conflicting_bad_signals_do_not_create_target(schema, config) -> None:
    result = apply(
        "БАД",
        "БАД в капсулах",
        "Также указано: спортивное питание",
        schema,
        config,
    )

    assert result.rule_id == "INSUFFICIENT_EVIDENCE"
    assert result.verdict is None
    assert result.conflict


def test_independent_ignition_source_is_detected(schema, config) -> None:
    result = apply(
        "Легковоспламеняющиеся",
        "Спички хозяйственные",
        "Предназначены для создания открытого огня",
        schema,
        config,
    )

    assert result.rule_id == "INDEPENDENT_IGNITION_SOURCE"
    assert result.verdict == 1
    assert result.values["independent_ignition_source"] is True
    assert result.values["object_type"] == "independent_ignition_source"


def test_lighter_accessory_does_not_become_ignition_source(schema, config) -> None:
    result = apply(
        "Легковоспламеняющиеся",
        "Чехол для зажигалки",
        "Защитный кожаный футляр для зажигалки",
        schema,
        config,
    )

    assert result.rule_id == "INSUFFICIENT_EVIDENCE"
    assert result.verdict is None
    assert result.values["independent_ignition_source"] == schema.unknown_value

    gas_cover = apply(
        "Легковоспламеняющиеся",
        "Чехол для газового баллона",
        "Сумка с подогревом для баллона",
        schema,
        config,
    )
    assert gas_cover.values["fuel_present"] == schema.unknown_value


def test_external_fuel_not_included_produces_negative_verdict(schema, config) -> None:
    result = apply(
        "Легковоспламеняющиеся",
        "Газовая горелка",
        "Работает от внешнего баллона. Баллон приобретается отдельно",
        schema,
        config,
    )

    assert result.rule_id == "DEVICE_WITHOUT_INCLUDED_FUEL"
    assert result.verdict == 0
    assert result.values["requires_external_fuel"] is True
    assert result.values["fuel_included"] is False
    assert result.values["fuel_present"] is False
    assert result.values["object_type"] == "device_requiring_external_fuel"

    empty_container = apply(
        "Легковоспламеняющиеся",
        "Газовый баллон 12 л (пустой)",
        "Пустой баллон поставляется без газа",
        schema,
        config,
    )
    assert empty_container.rule_id == "NO_FLAMMABLE_CONTENT"
    assert empty_container.verdict == 0
    assert empty_container.values["fuel_present"] is False


def test_embedded_ignition_and_component_rules_are_negative(schema, config) -> None:
    embedded = apply(
        "Легковоспламеняющиеся",
        "Газовая плита",
        "Оснащена встроенным электроподжигом",
        schema,
        config,
    )
    component = apply(
        "Легковоспламеняющиеся",
        "Фильтр для воды",
        "Картридж с активированным углём",
        schema,
        config,
    )

    assert embedded.rule_id == "DEVICE_WITH_EMBEDDED_IGNITION"
    assert embedded.verdict == 0
    assert embedded.values["independent_ignition_source"] is False
    assert component.rule_id == "FLAMMABLE_COMPONENT_ONLY"
    assert component.verdict == 0
    assert component.values["object_type"] == "flammable_component"


def test_conflicting_fuel_statements_remain_undecided(schema, config) -> None:
    result = apply(
        "Легковоспламеняющиеся",
        "Газ для зажигалок",
        "Баллон заправлен газом, но поставляется без газа",
        schema,
        config,
    )

    assert result.rule_id == "INSUFFICIENT_EVIDENCE"
    assert result.verdict is None
    assert result.conflict
    assert result.values["fuel_present"] == schema.unknown_value
    assert any(record.value is True for record in result.evidence)
    assert any(record.value is False for record in result.evidence)


def test_feature_builder_freezes_schema_and_evidence_json(schema, config) -> None:
    features = build_text_rule_features(
        sample_source(),
        sample_folds(),
        schema,
        config,
    )

    assert features.columns == list(TEXT_RULE_COLUMNS)
    assert dict(features.schema) == TEXT_RULE_SCHEMA
    assert features.height == 4
    assert features.get_column("id").to_list() == [1, 2, 3, 4]
    bad_row = features.filter(pl.col("id") == 1).row(0, named=True)
    evidence = json.loads(bad_row["text_evidence_json"])
    assert bad_row["text_explicit_bad_marker"] == 1
    assert bad_row["text_rule_verdict"] == 1
    assert evidence[0]["source"] == "description"


def test_rule_features_are_independent_of_true_label(schema, config) -> None:
    first = build_text_rule_features(
        sample_source(),
        sample_folds([1, 0, 1, 0]),
        schema,
        config,
    )
    second = build_text_rule_features(
        sample_source(),
        sample_folds([0, 1, 0, 1]),
        schema,
        config,
    )

    assert first.drop("true_label").equals(second.drop("true_label"))
    assert first.get_column("true_label").to_list() == [1, 0, 1, 0]
    assert second.get_column("true_label").to_list() == [0, 1, 0, 1]


def test_input_validation_rejects_category_mismatch() -> None:
    folds = sample_folds().with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit("Легковоспламеняющиеся"))
        .otherwise(pl.col("category"))
        .alias("category")
    )

    with pytest.raises(ValueError, match="categories differ"):
        validate_text_rule_inputs(sample_source(), folds)


def test_config_loader_rejects_invalid_regex(tmp_path) -> None:
    with open("configs/text_rules.yaml", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    invalid = deepcopy(payload)
    invalid["categories"]["БАД"]["signals"]["explicit_bad_marker"][
        "patterns"
    ] = ["("]
    path = tmp_path / "text_rules.yaml"
    path.write_text(
        yaml.safe_dump(invalid, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid regex"):
        load_text_rule_config(path)


def test_cli_writes_reproducible_feature_and_report_files(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.parquet"
    output_path = tmp_path / "text_rules.parquet"
    report_path = tmp_path / "report.md"
    source = sample_source().with_columns(
        pl.Series("label", [1, 0, 1, 0], dtype=pl.Int64)
    )
    source.write_csv(data_path)
    sample_folds().write_parquet(folds_path)

    exit_code = main(
        [
            "--data",
            str(data_path),
            "--folds",
            str(folds_path),
            "--schema",
            "configs/evidence.yaml",
            "--config",
            "configs/text_rules.yaml",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert pl.read_parquet(output_path).columns == list(TEXT_RULE_COLUMNS)
    report = report_path.read_text(encoding="utf-8")
    assert "# R09 — Text Rules" in report
    assert "Rule engine получает только" in report
