"""Shared, label-free primitives for deterministic text rules."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ecup.data.audit import EXPECTED_CATEGORIES
from ecup.rules.schema import EvidenceRecord, EvidenceSchema, EvidenceValue

DEFAULT_TEXT_RULES_PATH = Path("configs/text_rules.yaml")
SOURCE_ORDER = ("name", "description")
EXPECTED_SIGNALS = {
    "БАД": (
        "explicit_bad_marker",
        "dietary_supplement_marker",
        "explicit_not_bad",
        "sport_nutrition",
    ),
    "Легковоспламеняющиеся": (
        "independent_ignition_source",
        "fuel_present",
        "fuel_included",
        "requires_external_fuel",
        "flammable_item_in_kit",
        "flammable_item_not_in_kit",
        "explicit_without_fuel",
        "device_with_embedded_ignition",
        "flammable_component",
        "combustible_device",
    ),
}


@dataclass(frozen=True)
class SignalSpec:
    """Regexes that confirm one signal and nearby contexts that suppress it."""

    name: str
    patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]


@dataclass(frozen=True)
class CategoryRuleConfig:
    """Named signals available to one known competition category."""

    name: str
    signals: Mapping[str, SignalSpec]


@dataclass(frozen=True)
class TextRuleConfig:
    """Validated, versioned configuration for the text-only rule engine."""

    version: int
    exclusion_window: int
    negation_patterns: tuple[str, ...]
    categories: Mapping[str, CategoryRuleConfig]

    def category(self, category: str) -> CategoryRuleConfig:
        try:
            return self.categories[category]
        except KeyError as error:
            raise ValueError(f"unknown text-rule category: {category}") from error


@dataclass(frozen=True)
class TextSource:
    """Normalized text from one allowed source."""

    name: str
    text: str


@dataclass(frozen=True)
class TextDocument:
    """The only inputs visible to R09 rules."""

    sources: tuple[TextSource, ...]


@dataclass(frozen=True)
class TextMatch:
    """One concrete phrase matched in name or description."""

    source: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class RuleResult:
    """Category evidence and an optional conservative preliminary verdict."""

    category: str
    values: Mapping[str, EvidenceValue]
    evidence: tuple[EvidenceRecord, ...]
    rule_id: str
    verdict: int | None
    has_negation: bool
    conflict: bool
    matched_signals: tuple[str, ...]


def normalize_rule_text(value: object) -> str:
    """Normalize text without consulting labels or any image-derived data."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def prepare_document(name: object, description: object) -> TextDocument:
    """Build the fixed two-source input consumed by all R09 rules."""

    return TextDocument(
        sources=tuple(
            TextSource(source, normalize_rule_text(value))
            for source, value in (
                ("name", name),
                ("description", description),
            )
        )
    )


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


def _nearby(first: re.Match[str], second: re.Match[str], window: int) -> bool:
    return not (
        first.end() + window < second.start()
        or second.end() + window < first.start()
    )


def find_signal_matches(
    document: TextDocument,
    signal: SignalSpec,
    exclusion_window: int,
) -> tuple[TextMatch, ...]:
    """Find signal phrases, suppressing only nearby excluded contexts."""

    found: list[TextMatch] = []
    for source in document.sources:
        exclusions = [
            match
            for pattern in signal.exclude_patterns
            for match in _compiled(pattern).finditer(source.text)
        ]
        for pattern in signal.patterns:
            for match in _compiled(pattern).finditer(source.text):
                if any(
                    _nearby(match, exclusion, exclusion_window)
                    for exclusion in exclusions
                ):
                    continue
                found.append(
                    TextMatch(
                        source=source.name,
                        text=match.group(0).strip(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
    unique: dict[tuple[str, int, int, str], TextMatch] = {}
    for match in found:
        unique[(match.source, match.start, match.end, match.text)] = match
    source_rank = {source: index for index, source in enumerate(SOURCE_ORDER)}
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (source_rank[item.source], item.start, item.end),
        )
    )


def has_negation(document: TextDocument, config: TextRuleConfig) -> bool:
    return any(
        _compiled(pattern).search(source.text)
        for source in document.sources
        for pattern in config.negation_patterns
    )


def unknown_values(schema: EvidenceSchema, category: str) -> dict[str, str]:
    """Initialize every field explicitly to the R08 unknown sentinel."""

    return {
        field: schema.unknown_value
        for field in schema.category(category).fields
    }


def evidence_from_matches(
    schema: EvidenceSchema,
    category: str,
    field: str,
    value: EvidenceValue,
    matches: Sequence[TextMatch],
) -> tuple[EvidenceRecord, ...]:
    """Convert concrete regex matches into validated R08 evidence records."""

    records: list[EvidenceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for match in matches:
        key = (match.source, match.text, repr(value))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            schema.parse_evidence(
                category,
                {
                    "field": field,
                    "value": value,
                    "source": match.source,
                    "text": match.text,
                },
            )
        )
    return tuple(records)


def validate_rule_result(
    result: RuleResult,
    schema: EvidenceSchema,
) -> RuleResult:
    """Enforce the R08 schema and conservative verdict invariants."""

    schema.validate_values(result.category, result.values, require_all=True)
    schema.validate_rule_id(result.category, result.rule_id)
    if result.verdict is not None and result.verdict not in (0, 1):
        raise ValueError("rule verdict must be 0, 1 or None")
    if result.conflict and result.verdict is not None:
        raise ValueError("conflicting evidence must not produce a verdict")
    if result.verdict is None and result.rule_id != "INSUFFICIENT_EVIDENCE":
        raise ValueError("missing verdict must use INSUFFICIENT_EVIDENCE")
    for record in result.evidence:
        schema.parse_evidence(result.category, record.to_dict())
    return result


def _as_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _exact_keys(
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


def _patterns(value: object, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    patterns = tuple(value)
    if (not patterns and not allow_empty) or not all(
        isinstance(pattern, str) and pattern for pattern in patterns
    ):
        raise ValueError(f"{path} must contain non-empty regex strings")
    if len(set(patterns)) != len(patterns):
        raise ValueError(f"{path} contains duplicate regexes")
    for pattern in patterns:
        try:
            _compiled(pattern)
        except re.error as error:
            raise ValueError(f"invalid regex in {path}: {pattern}") from error
    return patterns


def load_text_rule_config(
    path: str | Path = DEFAULT_TEXT_RULES_PATH,
) -> TextRuleConfig:
    """Load and strictly validate the versioned R09 pattern configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _as_mapping(payload, "text_rules")
    _exact_keys(
        root,
        {"version", "exclusion_window", "negation_patterns", "categories"},
        "text_rules",
    )
    version = root["version"]
    window = root["exclusion_window"]
    if type(version) is not int or version != 1:
        raise ValueError("text_rules.version must be integer 1")
    if type(window) is not int or window < 0:
        raise ValueError(
            "text_rules.exclusion_window must be a non-negative integer"
        )
    negations = _patterns(
        root["negation_patterns"],
        "text_rules.negation_patterns",
        allow_empty=False,
    )

    raw_categories = _as_mapping(root["categories"], "text_rules.categories")
    if set(raw_categories) != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "text_rules.categories must be exactly: "
            + ", ".join(EXPECTED_CATEGORIES)
        )
    categories: dict[str, CategoryRuleConfig] = {}
    for category in EXPECTED_CATEGORIES:
        category_path = f"text_rules.categories.{category}"
        raw_category = _as_mapping(raw_categories[category], category_path)
        _exact_keys(raw_category, {"signals"}, category_path)
        raw_signals = _as_mapping(
            raw_category["signals"],
            f"{category_path}.signals",
        )
        if set(raw_signals) != set(EXPECTED_SIGNALS[category]):
            raise ValueError(
                f"{category_path}.signals must be exactly: "
                + ", ".join(EXPECTED_SIGNALS[category])
            )
        signals: dict[str, SignalSpec] = {}
        for signal_name in EXPECTED_SIGNALS[category]:
            signal_path = f"{category_path}.signals.{signal_name}"
            raw_signal = _as_mapping(raw_signals[signal_name], signal_path)
            _exact_keys(
                raw_signal,
                {"patterns", "exclude_patterns"},
                signal_path,
            )
            signals[signal_name] = SignalSpec(
                name=signal_name,
                patterns=_patterns(
                    raw_signal["patterns"],
                    f"{signal_path}.patterns",
                    allow_empty=False,
                ),
                exclude_patterns=_patterns(
                    raw_signal["exclude_patterns"],
                    f"{signal_path}.exclude_patterns",
                    allow_empty=True,
                ),
            )
        categories[category] = CategoryRuleConfig(
            name=category,
            signals=signals,
        )
    return TextRuleConfig(
        version=version,
        exclusion_window=window,
        negation_patterns=negations,
        categories=categories,
    )
