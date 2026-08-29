"""Build deterministic Text Rule features from name and description only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ecup.contracts import OOF_BASE_COLUMNS
from ecup.data.audit import EXPECTED_CATEGORIES, load_dataset
from ecup.rules.bad import apply_bad_rules
from ecup.rules.engine import (
    DEFAULT_TEXT_RULES_PATH,
    RuleResult,
    TextRuleConfig,
    load_text_rule_config,
    prepare_document,
)
from ecup.rules.flammable import apply_flammable_rules
from ecup.rules.schema import DEFAULT_SCHEMA_PATH, EvidenceSchema, load_evidence_schema

DEFAULT_DATA_PATH = Path("data/raw/data.csv")
DEFAULT_FOLDS_PATH = Path("data/processed/folds.parquet")
DEFAULT_OUTPUT_PATH = Path("features/text_rules.parquet")
DEFAULT_REPORT_PATH = Path("reports/R09-text-rules.md")

SOURCE_COLUMNS = ("id", "name", "description", "category")
FOLD_INPUT_COLUMNS = ("id", "category", "label", "group_id", "fold")
TRI_STATE_FEATURES = (
    "text_explicit_bad_marker",
    "text_dietary_supplement_marker",
    "text_explicit_not_bad",
    "text_sport_nutrition",
    "text_fuel_present",
    "text_fuel_included",
    "text_requires_external_fuel",
    "text_independent_ignition_source",
    "text_flammable_item_in_kit",
    "text_explicit_without_fuel",
)
BOOLEAN_SIGNAL_FEATURES = ("text_combustible_device",)
MARKER_SOURCE_VALUES = ("name", "description", "unknown")
OBJECT_TYPE_VALUES = (
    "independent_ignition_source",
    "combustible_content",
    "device_requiring_external_fuel",
    "device_with_embedded_ignition",
    "flammable_component",
    "kit",
    "other",
    "unknown",
)
MARKER_SOURCE_FEATURES = tuple(
    f"text_marker_source_{value}" for value in MARKER_SOURCE_VALUES
)
OBJECT_TYPE_FEATURES = tuple(
    f"text_object_type_{value}" for value in OBJECT_TYPE_VALUES
)
RULE_META_COLUMNS = (
    "text_rule_id",
    "text_rule_verdict",
    "text_rule_has_verdict",
    "text_rule_conflict",
    "text_has_negation",
    "text_evidence_count",
    "text_name_evidence_count",
    "text_description_evidence_count",
    "text_matched_signals_json",
    "text_values_json",
    "text_evidence_json",
)
TEXT_RULE_COLUMNS = (
    *OOF_BASE_COLUMNS,
    *TRI_STATE_FEATURES,
    *BOOLEAN_SIGNAL_FEATURES,
    *MARKER_SOURCE_FEATURES,
    *OBJECT_TYPE_FEATURES,
    *RULE_META_COLUMNS,
)
TEXT_RULE_SCHEMA = {
    "id": pl.Int64,
    "category": pl.String,
    "fold": pl.Int8,
    "group_id": pl.String,
    "true_label": pl.Int64,
    **{column: pl.Int8 for column in TRI_STATE_FEATURES},
    **{column: pl.Int8 for column in BOOLEAN_SIGNAL_FEATURES},
    **{column: pl.Int8 for column in MARKER_SOURCE_FEATURES},
    **{column: pl.Int8 for column in OBJECT_TYPE_FEATURES},
    "text_rule_id": pl.String,
    "text_rule_verdict": pl.Int8,
    "text_rule_has_verdict": pl.Boolean,
    "text_rule_conflict": pl.Boolean,
    "text_has_negation": pl.Boolean,
    "text_evidence_count": pl.UInt16,
    "text_name_evidence_count": pl.UInt16,
    "text_description_evidence_count": pl.UInt16,
    "text_matched_signals_json": pl.String,
    "text_values_json": pl.String,
    "text_evidence_json": pl.String,
}


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


def validate_text_rule_inputs(source: pl.DataFrame, folds: pl.DataFrame) -> None:
    """Validate joins without exposing label to the category rule functions."""

    _require_columns(source, SOURCE_COLUMNS, "source")
    _require_columns(folds, FOLD_INPUT_COLUMNS, "folds")
    if source.is_empty():
        raise ValueError("source dataset is empty")
    for frame, name in ((source, "source"), (folds, "folds")):
        if frame.get_column("id").null_count():
            raise ValueError(f"{name} id contains null values")
        if frame.get_column("id").n_unique() != frame.height:
            raise ValueError(f"{name} id values must be unique")
    if source.height != folds.height:
        raise ValueError("source and folds row counts differ")
    joined = source.select("id", "category").join(
        folds.select("id", "category"),
        on="id",
        how="inner",
        suffix="_fold",
    )
    if joined.height != source.height:
        raise ValueError("source and folds id sets differ")
    if joined.filter(pl.col("category") != pl.col("category_fold")).height:
        raise ValueError("source and folds categories differ")
    categories = set(source.get_column("category").drop_nulls().to_list())
    if categories != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "source categories must be exactly: " + ", ".join(EXPECTED_CATEGORIES)
        )


def apply_text_rules(
    name: object,
    description: object,
    category: str,
    schema: EvidenceSchema,
    config: TextRuleConfig,
) -> RuleResult:
    """Dispatch to a category rule set using only the permitted text fields."""

    document = prepare_document(name, description)
    if category == "БАД":
        return apply_bad_rules(document, schema, config)
    if category == "Легковоспламеняющиеся":
        return apply_flammable_rules(document, schema, config)
    raise ValueError(f"unknown text-rule category: {category}")


def _tri_state(value: object, unknown_value: str) -> int:
    if value == unknown_value:
        return -1
    if type(value) is bool:
        return int(value)
    raise ValueError(f"cannot encode tri-state value: {value}")


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_features(result: RuleResult, schema: EvidenceSchema) -> dict[str, object]:
    values = result.values
    row: dict[str, object] = {
        column: -1 for column in TRI_STATE_FEATURES
    }
    row["text_combustible_device"] = int(
        "combustible_device" in result.matched_signals
    )
    if result.category == "БАД":
        for field in (
            "explicit_bad_marker",
            "dietary_supplement_marker",
            "explicit_not_bad",
            "sport_nutrition",
        ):
            row[f"text_{field}"] = _tri_state(
                values[field],
                schema.unknown_value,
            )
    else:
        for field in (
            "fuel_present",
            "fuel_included",
            "requires_external_fuel",
            "independent_ignition_source",
            "flammable_item_in_kit",
            "explicit_without_fuel",
        ):
            row[f"text_{field}"] = _tri_state(
                values[field],
                schema.unknown_value,
            )

    marker_source = (
        values["marker_source"]
        if result.category == "БАД"
        else None
    )
    row.update(
        {
            f"text_marker_source_{value}": int(marker_source == value)
            for value in MARKER_SOURCE_VALUES
        }
    )
    object_type = (
        values["object_type"]
        if result.category == "Легковоспламеняющиеся"
        else None
    )
    row.update(
        {
            f"text_object_type_{value}": int(object_type == value)
            for value in OBJECT_TYPE_VALUES
        }
    )

    evidence_payload = [record.to_dict() for record in result.evidence]
    row.update(
        text_rule_id=result.rule_id,
        text_rule_verdict=result.verdict,
        text_rule_has_verdict=result.verdict is not None,
        text_rule_conflict=result.conflict,
        text_has_negation=result.has_negation,
        text_evidence_count=len(result.evidence),
        text_name_evidence_count=sum(
            record.source == "name" for record in result.evidence
        ),
        text_description_evidence_count=sum(
            record.source == "description" for record in result.evidence
        ),
        text_matched_signals_json=_stable_json(result.matched_signals),
        text_values_json=_stable_json(dict(result.values)),
        text_evidence_json=_stable_json(evidence_payload),
    )
    return row


def validate_text_rule_features(
    source: pl.DataFrame,
    folds: pl.DataFrame,
    features: pl.DataFrame,
    schema: EvidenceSchema,
) -> None:
    """Freeze the R09 feature schema and all row-level alignment invariants."""

    if features.columns != list(TEXT_RULE_COLUMNS):
        raise ValueError(f"text rule columns must be exactly: {TEXT_RULE_COLUMNS}")
    if dict(features.schema) != TEXT_RULE_SCHEMA:
        raise ValueError("text rule dtypes do not match the frozen schema")
    if features.height != source.height:
        raise ValueError("text rules must contain exactly one row per source row")
    if features.get_column("id").n_unique() != features.height:
        raise ValueError("text rule id values must be unique")
    expected_base = (
        folds.select(
            "id",
            "category",
            "fold",
            "group_id",
            pl.col("label").alias("true_label"),
        )
        .sort("id")
    )
    if not expected_base.equals(features.select(OOF_BASE_COLUMNS).sort("id")):
        raise ValueError("text rule base columns differ from folds")
    if features.filter(
        pl.col("text_rule_conflict") & pl.col("text_rule_has_verdict")
    ).height:
        raise ValueError("conflicting rules must not produce verdicts")
    if features.filter(
        pl.col("text_rule_has_verdict")
        != pl.col("text_rule_verdict").is_not_null()
    ).height:
        raise ValueError("text_rule_has_verdict disagrees with verdict nullability")
    if features.filter(
        pl.col("text_rule_verdict").is_not_null()
        & ~pl.col("text_rule_verdict").is_in([0, 1])
    ).height:
        raise ValueError("text_rule_verdict must contain only 0, 1 or null")
    if features.filter(
        pl.col("text_evidence_count")
        != pl.col("text_name_evidence_count")
        + pl.col("text_description_evidence_count")
    ).height:
        raise ValueError("text evidence source counts do not add up")

    for row in features.select(
        "category",
        "text_rule_id",
        "text_rule_verdict",
        "text_rule_has_verdict",
        "text_values_json",
        "text_evidence_json",
    ).iter_rows(named=True):
        values = json.loads(row["text_values_json"])
        schema.validate_values(row["category"], values, require_all=True)
        schema.validate_rule_id(row["category"], row["text_rule_id"])
        evidence = json.loads(row["text_evidence_json"])
        schema.parse_evidence_list(row["category"], evidence)
        if row["text_rule_has_verdict"]:
            schema.build_target(
                row["category"],
                values,
                row["text_rule_id"],
                row["text_rule_verdict"],
            )


def build_text_rule_features(
    source: pl.DataFrame,
    folds: pl.DataFrame,
    schema: EvidenceSchema,
    config: TextRuleConfig,
) -> pl.DataFrame:
    """Build R09 features, joining true labels only after rule evaluation."""

    validate_text_rule_inputs(source, folds)
    rule_rows = []
    for row in source.select(SOURCE_COLUMNS).sort("id").iter_rows(named=True):
        result = apply_text_rules(
            row["name"],
            row["description"],
            row["category"],
            schema,
            config,
        )
        rule_rows.append({"id": row["id"], **_result_features(result, schema)})

    rule_frame = pl.DataFrame(rule_rows).with_columns(
        *[
            pl.col(column).cast(dtype)
            for column, dtype in TEXT_RULE_SCHEMA.items()
            if column not in OOF_BASE_COLUMNS
        ]
    )
    features = (
        folds.select(
            "id",
            "category",
            "fold",
            "group_id",
            pl.col("label").alias("true_label"),
        )
        .join(rule_frame, on="id", how="inner", validate="1:1")
        .select(TEXT_RULE_COLUMNS)
        .sort("id")
    )
    validate_text_rule_features(source, folds, features, schema)
    return features


def feature_checksum(features: pl.DataFrame) -> str:
    """Hash logical R09 rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in features.select(TEXT_RULE_COLUMNS).sort("id").iter_rows():
        digest.update(_stable_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "0.00%"


def render_text_rule_report(
    features: pl.DataFrame,
    data_path: str | Path,
    folds_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    config: TextRuleConfig,
) -> str:
    """Render coverage and label-only diagnostics for the deterministic rules."""

    lines = [
        "# R09 — Text Rules",
        "",
        "## Контракт",
        "",
        f"- Данные: `{data_path}`.",
        f"- Folds: `{folds_path}`.",
        f"- Конфигурация: `{config_path}`, версия **{config.version}**.",
        f"- Артефакт: `{output_path}`.",
        f"- Строк: **{features.height}**.",
        f"- Колонок: **{len(features.columns)}**.",
        f"- Логическая SHA-256: `{feature_checksum(features)}`.",
        "- Rule engine получает только `name`, `description` и известную `category`.",
        (
            "- `true_label` присоединяется после правил и используется "
            "только для отчётной проверки."
        ),
        "- OCR и изображения в R09 не используются.",
        (
            "- `unknown` кодируется как `-1` в tri-state feature, "
            "а отсутствие verdict — как `null`."
        ),
        "",
        "## Покрытие предварительными verdict",
        "",
        (
            "| Категория | Всего | Есть verdict | Покрытие | "
            "Верно на покрытых | Accuracy на покрытых | Конфликты |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in EXPECTED_CATEGORIES:
        subset = features.filter(pl.col("category") == category)
        decided = subset.filter(pl.col("text_rule_has_verdict"))
        correct = decided.filter(
            pl.col("text_rule_verdict") == pl.col("true_label")
        ).height
        conflicts = subset.filter(pl.col("text_rule_conflict")).height
        lines.append(
            f"| {category} | {subset.height} | {decided.height} | "
            f"{_percent(decided.height, subset.height)} | {correct} | "
            f"{_percent(correct, decided.height)} | {conflicts} |"
        )

    distribution = (
        features.group_by("category", "text_rule_id")
        .agg(pl.len().alias("count"))
        .sort("category", "count", descending=[False, True])
    )
    lines.extend(
        [
            "",
            "## Распределение rule_id",
            "",
            "| Категория | rule_id | Количество |",
            "|---|---|---:|",
        ]
    )
    for row in distribution.iter_rows(named=True):
        lines.append(
            f"| {row['category']} | `{row['text_rule_id']}` | {row['count']} |"
        )

    verdict_quality = (
        features.filter(pl.col("text_rule_has_verdict"))
        .group_by("category", "text_rule_verdict")
        .agg(
            pl.len().alias("count"),
            (pl.col("text_rule_verdict") == pl.col("true_label"))
            .sum()
            .alias("correct"),
        )
        .sort("category", "text_rule_verdict")
    )
    lines.extend(
        [
            "",
            "## Диагностика по полярности rule verdict",
            "",
            "| Категория | Rule verdict | Количество | Совпало с label | Accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in verdict_quality.iter_rows(named=True):
        lines.append(
            f"| {row['category']} | {row['text_rule_verdict']} | "
            f"{row['count']} | {row['correct']} | "
            f"{_percent(row['correct'], row['count'])} |"
        )

    lines.extend(
        [
            "",
            "## Границы этапа",
            "",
            (
                "R09 создаёт проверяемые label-free признаки и "
                "предварительные rule verdict. Он не обучает "
                "TF-IDF и не создаёт OOF-предсказания — это выполняется в R10."
            ),
            "",
            (
                "Конфликтующие и недостаточные случаи сохраняются с "
                "`INSUFFICIENT_EVIDENCE`; исходный label не используется "
                "для заполнения evidence."
            ),
            "",
            (
                "Rule verdict не является финальным классификатором и не "
                "становится silver-target автоматически. Низкая точность "
                "положительных текстовых сигналов для воспламеняемости "
                "подтверждает необходимость OCR и изображений."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build label-free Text Rule features for E-CUP"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_TEXT_RULES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_dataset(args.data)
    folds = pl.read_parquet(args.folds)
    schema = load_evidence_schema(args.schema)
    config = load_text_rule_config(args.config)
    features = build_text_rule_features(source, folds, schema, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(args.output, compression="zstd")
    report = render_text_rule_report(
        features,
        args.data,
        args.folds,
        args.config,
        args.output,
        config,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        f"Text Rules complete: {features.height} rows, "
        f"{int(features.get_column('text_rule_has_verdict').sum())} verdicts, "
        f"checksum={feature_checksum(features)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
