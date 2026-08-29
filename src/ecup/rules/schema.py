"""Frozen evidence contracts shared by rules, LoRA and explanations."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ecup.data.audit import EXPECTED_CATEGORIES

DEFAULT_SCHEMA_PATH = Path("configs/evidence.yaml")
FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
RULE_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
EXPECTED_SOURCES = ("name", "description", "ocr", "image")
FIELD_KINDS = frozenset({"tri_state", "enum"})

EvidenceValue = bool | str


@dataclass(frozen=True)
class SourceSpec:
    """Location requirements for one evidence source."""

    name: str
    requires_text: bool
    requires_image_index: bool


@dataclass(frozen=True)
class FieldSpec:
    """Allowed values and documentation for one category field."""

    name: str
    kind: str
    description: str
    allowed_values: tuple[str, ...] = ()

    def validate(self, value: object, unknown_value: str) -> EvidenceValue:
        if self.kind == "tri_state":
            if type(value) is bool:
                return value
            if isinstance(value, str) and value == unknown_value:
                return value
            raise ValueError(
                f"field {self.name} expects true, false or {unknown_value}"
            )
        if self.kind == "enum":
            if isinstance(value, str) and value in self.allowed_values:
                return value
            raise ValueError(
                f"field {self.name} expects one of: "
                + ", ".join(self.allowed_values)
            )
        raise RuntimeError(f"unsupported field kind: {self.kind}")


@dataclass(frozen=True)
class CategorySpec:
    """Evidence fields and rule identifiers for one competition category."""

    name: str
    fields: Mapping[str, FieldSpec]
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceRecord:
    """One confirmed fact and the location that supports it."""

    field: str
    value: EvidenceValue
    source: str
    text: str | None = None
    image_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "field": self.field,
            "value": self.value,
            "source": self.source,
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.image_index is not None:
            payload["image_index"] = self.image_index
        return payload


@dataclass(frozen=True)
class EvidenceSchema:
    """Validated, category-aware evidence contract."""

    version: int
    unknown_value: str
    sources: Mapping[str, SourceSpec]
    categories: Mapping[str, CategorySpec]

    def category(self, category: str) -> CategorySpec:
        try:
            return self.categories[category]
        except KeyError as error:
            raise ValueError(f"unknown evidence category: {category}") from error

    def field(self, category: str, field: str) -> FieldSpec:
        category_spec = self.category(category)
        try:
            return category_spec.fields[field]
        except KeyError as error:
            raise ValueError(
                f"unknown evidence field for {category}: {field}"
            ) from error

    def validate_value(
        self,
        category: str,
        field: str,
        value: object,
    ) -> EvidenceValue:
        return self.field(category, field).validate(value, self.unknown_value)

    def validate_rule_id(self, category: str, rule_id: object) -> str:
        category_spec = self.category(category)
        if not isinstance(rule_id, str) or rule_id not in category_spec.rule_ids:
            raise ValueError(f"unknown rule_id for {category}: {rule_id}")
        return rule_id

    def parse_evidence(
        self,
        category: str,
        payload: Mapping[str, object],
    ) -> EvidenceRecord:
        """Validate one concrete fact; unknown belongs in targets, not evidence."""

        if not isinstance(payload, Mapping):
            raise ValueError("evidence must be a mapping")
        required = {"field", "value", "source"}
        allowed = required | {"text", "image_index"}
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - allowed)
        if missing:
            raise ValueError("evidence misses fields: " + ", ".join(missing))
        if extra:
            raise ValueError("evidence has unknown fields: " + ", ".join(extra))

        field = payload["field"]
        source = payload["source"]
        if not isinstance(field, str):
            raise ValueError("evidence field must be a string")
        if not isinstance(source, str) or source not in self.sources:
            raise ValueError(f"unknown evidence source: {source}")
        value = self.validate_value(category, field, payload["value"])
        if value == self.unknown_value:
            raise ValueError("unknown is not a confirmed evidence fact")

        source_spec = self.sources[source]
        raw_text = payload.get("text")
        text: str | None = None
        if raw_text is not None:
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("evidence text must be a non-empty string")
            text = raw_text.strip()
        if source_spec.requires_text and text is None:
            raise ValueError(f"source {source} requires evidence text")

        raw_image_index = payload.get("image_index")
        image_index: int | None = None
        if raw_image_index is not None:
            if type(raw_image_index) is not int or raw_image_index < 0:
                raise ValueError("image_index must be a non-negative integer")
            image_index = raw_image_index
        if source_spec.requires_image_index and image_index is None:
            raise ValueError(f"source {source} requires image_index")
        if not source_spec.requires_image_index and image_index is not None:
            raise ValueError(f"source {source} must not contain image_index")

        return EvidenceRecord(
            field=field,
            value=value,
            source=source,
            text=text,
            image_index=image_index,
        )

    def parse_evidence_list(
        self,
        category: str,
        payloads: Sequence[Mapping[str, object]],
    ) -> tuple[EvidenceRecord, ...]:
        if isinstance(payloads, (str, bytes)) or not isinstance(
            payloads, Sequence
        ):
            raise ValueError("evidence list must be a sequence")
        return tuple(self.parse_evidence(category, item) for item in payloads)

    def validate_values(
        self,
        category: str,
        values: Mapping[str, object],
        *,
        require_all: bool = True,
    ) -> dict[str, EvidenceValue]:
        """Validate flat category fields used by rules and LoRA targets."""

        if not isinstance(values, Mapping):
            raise ValueError("evidence values must be a mapping")
        category_spec = self.category(category)
        expected = set(category_spec.fields)
        received = set(values)
        extra = sorted(received - expected)
        missing = sorted(expected - received) if require_all else []
        if extra:
            raise ValueError("unknown evidence value fields: " + ", ".join(extra))
        if missing:
            raise ValueError("evidence values miss fields: " + ", ".join(missing))

        validated: dict[str, EvidenceValue] = {}
        for name in category_spec.fields:
            if name in values:
                validated[name] = self.validate_value(
                    category,
                    name,
                    values[name],
                )
        return validated

    def build_target(
        self,
        category: str,
        values: Mapping[str, object],
        rule_id: object,
        verdict: object,
    ) -> dict[str, object]:
        """Build the strict flat JSON target consumed by LoRA datasets."""

        validated = self.validate_values(category, values, require_all=True)
        validated_rule_id = self.validate_rule_id(category, rule_id)
        if type(verdict) is not int or verdict not in (0, 1):
            raise ValueError("verdict must be integer 0 or 1")
        return {
            **validated,
            "rule_id": validated_rule_id,
            "verdict": verdict,
        }

    def validate_target(
        self,
        category: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(payload, Mapping):
            raise ValueError("target must be a mapping")
        category_spec = self.category(category)
        field_names = set(category_spec.fields)
        allowed = field_names | {"rule_id", "verdict"}
        extra = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if extra:
            raise ValueError("target has unknown fields: " + ", ".join(extra))
        if missing:
            raise ValueError("target misses fields: " + ", ".join(missing))
        return self.build_target(
            category,
            {name: payload[name] for name in category_spec.fields},
            payload["rule_id"],
            payload["verdict"],
        )


def _as_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    extra = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if extra:
        raise ValueError(f"{path} has unknown keys: {', '.join(extra)}")
    if missing:
        raise ValueError(f"{path} misses keys: {', '.join(missing)}")


def _load_sources(payload: object) -> dict[str, SourceSpec]:
    source_payload = _as_mapping(payload, "sources")
    if set(source_payload) != set(EXPECTED_SOURCES):
        raise ValueError(
            "sources must be exactly: " + ", ".join(EXPECTED_SOURCES)
        )
    sources: dict[str, SourceSpec] = {}
    for name in EXPECTED_SOURCES:
        raw_spec = source_payload[name]
        if not FIELD_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid source name: {name}")
        spec = _as_mapping(raw_spec, f"sources.{name}")
        _require_exact_keys(
            spec,
            {"requires_text", "requires_image_index"},
            f"sources.{name}",
        )
        requires_text = spec["requires_text"]
        requires_image_index = spec["requires_image_index"]
        if type(requires_text) is not bool or type(requires_image_index) is not bool:
            raise ValueError(f"sources.{name} requirements must be booleans")
        sources[name] = SourceSpec(
            name=name,
            requires_text=requires_text,
            requires_image_index=requires_image_index,
        )
    return sources


def _load_field(
    name: str,
    payload: object,
    unknown_value: str,
    path: str,
) -> FieldSpec:
    if not FIELD_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid evidence field name: {name}")
    spec = _as_mapping(payload, path)
    allowed_keys = {"type", "description", "allowed_values"}
    extra = sorted(set(spec) - allowed_keys)
    missing = sorted({"type", "description"} - set(spec))
    if extra:
        raise ValueError(f"{path} has unknown keys: {', '.join(extra)}")
    if missing:
        raise ValueError(f"{path} misses keys: {', '.join(missing)}")
    kind = spec["type"]
    description = spec["description"]
    if not isinstance(kind, str) or kind not in FIELD_KINDS:
        raise ValueError(f"{path}.type must be tri_state or enum")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}.description must be non-empty")

    raw_values = spec.get("allowed_values")
    allowed_values: tuple[str, ...] = ()
    if kind == "tri_state" and raw_values is not None:
        raise ValueError(f"{path} tri_state must not define allowed_values")
    if kind == "enum":
        if isinstance(raw_values, (str, bytes)) or not isinstance(
            raw_values, Sequence
        ):
            raise ValueError(f"{path}.allowed_values must be a sequence")
        if not raw_values or not all(isinstance(item, str) for item in raw_values):
            raise ValueError(f"{path}.allowed_values must contain strings")
        allowed_values = tuple(raw_values)
        if len(set(allowed_values)) != len(allowed_values):
            raise ValueError(f"{path}.allowed_values contains duplicates")
        if unknown_value not in allowed_values:
            raise ValueError(f"{path}.allowed_values must contain {unknown_value}")

    return FieldSpec(
        name=name,
        kind=kind,
        description=description.strip(),
        allowed_values=allowed_values,
    )


def _load_categories(
    payload: object,
    unknown_value: str,
) -> dict[str, CategorySpec]:
    category_payload = _as_mapping(payload, "categories")
    expected_categories = set(EXPECTED_CATEGORIES)
    if set(category_payload) != expected_categories:
        raise ValueError(
            "categories must be exactly: " + ", ".join(EXPECTED_CATEGORIES)
        )
    categories: dict[str, CategorySpec] = {}
    for category in EXPECTED_CATEGORIES:
        path = f"categories.{category}"
        spec = _as_mapping(category_payload[category], path)
        _require_exact_keys(spec, {"fields", "rule_ids"}, path)
        field_payload = _as_mapping(spec["fields"], f"{path}.fields")
        if not field_payload:
            raise ValueError(f"{path}.fields must not be empty")
        fields = {
            name: _load_field(
                name,
                raw_field,
                unknown_value,
                f"{path}.fields.{name}",
            )
            for name, raw_field in field_payload.items()
        }

        rule_ids = spec["rule_ids"]
        if isinstance(rule_ids, (str, bytes)) or not isinstance(rule_ids, Sequence):
            raise ValueError(f"{path}.rule_ids must be a sequence")
        if not rule_ids or not all(isinstance(item, str) for item in rule_ids):
            raise ValueError(f"{path}.rule_ids must contain strings")
        normalized_rule_ids = tuple(rule_ids)
        if len(set(normalized_rule_ids)) != len(normalized_rule_ids):
            raise ValueError(f"{path}.rule_ids contains duplicates")
        if not all(RULE_ID_PATTERN.fullmatch(item) for item in normalized_rule_ids):
            raise ValueError(f"{path}.rule_ids must use UPPER_SNAKE_CASE")
        categories[category] = CategorySpec(
            name=category,
            fields=fields,
            rule_ids=normalized_rule_ids,
        )
    return categories


def load_evidence_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> EvidenceSchema:
    """Load and structurally validate the canonical YAML contract."""

    schema_path = Path(path)
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    root = _as_mapping(payload, "schema")
    _require_exact_keys(
        root,
        {"version", "unknown_value", "sources", "categories"},
        "schema",
    )
    version = root["version"]
    unknown_value = root["unknown_value"]
    if type(version) is not int or version != 1:
        raise ValueError("schema.version must be integer 1")
    if not isinstance(unknown_value, str) or not unknown_value:
        raise ValueError("schema.unknown_value must be a non-empty string")
    return EvidenceSchema(
        version=version,
        unknown_value=unknown_value,
        sources=_load_sources(root["sources"]),
        categories=_load_categories(root["categories"], unknown_value),
    )


def _json_block(payload: object) -> str:
    return "```json\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n```"


def render_schema_report(schema: EvidenceSchema, schema_path: str | Path) -> str:
    """Render the tracked R08 contract documentation from the YAML schema."""

    lines = [
        "# R08 — evidence schema",
        "",
        "## Контракт",
        "",
        f"- Источник схемы: `{schema_path}`.",
        f"- Версия: **{schema.version}**.",
        f"- Значение недостатка информации: `{schema.unknown_value}`.",
        "- `unknown` допустим в полях target, но не создаётся как подтверждённый evidence.",
        "- Исходный `label` не входит в evidence и не может подтверждать факт.",
        "",
        "## Источники",
        "",
        "| Source | Нужен text | Нужен image_index |",
        "|---|---:|---:|",
    ]
    for source in schema.sources.values():
        lines.append(
            f"| `{source.name}` | {str(source.requires_text).lower()} | "
            f"{str(source.requires_image_index).lower()} |"
        )

    for category in EXPECTED_CATEGORIES:
        category_spec = schema.category(category)
        lines.extend(
            [
                "",
                f"## Категория: {category}",
                "",
                "| Поле | Тип | Допустимые значения | Описание |",
                "|---|---|---|---|",
            ]
        )
        for field in category_spec.fields.values():
            allowed = (
                "true / false / unknown"
                if field.kind == "tri_state"
                else " / ".join(field.allowed_values)
            )
            lines.append(
                f"| `{field.name}` | `{field.kind}` | {allowed} | "
                f"{field.description} |"
            )
        lines.extend(
            [
                "",
                "Допустимые `rule_id`: "
                + ", ".join(f"`{rule_id}`" for rule_id in category_spec.rule_ids)
                + ".",
            ]
        )

    bad_values = {
        name: schema.unknown_value
        for name in schema.category("БАД").fields
    }
    bad_values.update(
        explicit_bad_marker=True,
        marker_source="description",
    )
    flammable_values = {
        name: schema.unknown_value
        for name in schema.category("Легковоспламеняющиеся").fields
    }
    flammable_values.update(
        object_type="device_requiring_external_fuel",
        fuel_included=False,
        requires_external_fuel=True,
        explicit_without_fuel=True,
    )
    bad_evidence = schema.parse_evidence(
        "БАД",
        {
            "field": "explicit_bad_marker",
            "value": True,
            "source": "description",
            "text": "Биологически активная добавка к пище",
        },
    )
    flammable_evidence = schema.parse_evidence(
        "Легковоспламеняющиеся",
        {
            "field": "fuel_included",
            "value": False,
            "source": "description",
            "text": "Баллон приобретается отдельно",
        },
    )
    lines.extend(
        [
            "",
            "## Примеры evidence",
            "",
            "### БАД",
            "",
            _json_block(bad_evidence.to_dict()),
            "",
            "### Легковоспламеняющиеся",
            "",
            _json_block(flammable_evidence.to_dict()),
            "",
            "## Примеры LoRA target",
            "",
            "### БАД",
            "",
            _json_block(
                schema.build_target(
                    "БАД",
                    bad_values,
                    "EXPLICIT_BAD_MARKER",
                    1,
                )
            ),
            "",
            "### Легковоспламеняющиеся",
            "",
            _json_block(
                schema.build_target(
                    "Легковоспламеняющиеся",
                    flammable_values,
                    "DEVICE_WITHOUT_INCLUDED_FUEL",
                    0,
                )
            ),
            "",
            "## Границы R08",
            "",
            "R08 фиксирует только имена, типы, источники и сериализацию. Словари, регулярные выражения, отрицания и условия `if/else` реализуются в R09 и R12.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and document the E-CUP evidence schema"
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default="reports/R08-evidence-schema.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = load_evidence_schema(args.schema)
    report = render_schema_report(schema, args.schema)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        f"Evidence schema complete: version={schema.version}, "
        f"categories={len(schema.categories)}, sources={len(schema.sources)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
