"""Text-only evidence rules for the flammable-products category."""

from __future__ import annotations

from collections.abc import Sequence

from ecup.rules.engine import (
    RuleResult,
    TextDocument,
    TextMatch,
    TextRuleConfig,
    evidence_from_matches,
    find_signal_matches,
    has_negation,
    unknown_values,
    validate_rule_result,
)
from ecup.rules.schema import EvidenceSchema

CATEGORY = "Легковоспламеняющиеся"


def _set_boolean(
    values: dict[str, object],
    evidence: list,
    schema: EvidenceSchema,
    field: str,
    true_matches: Sequence[TextMatch] = (),
    false_matches: Sequence[TextMatch] = (),
) -> bool:
    """Set one tri-state field, preserving contradictory evidence as unknown."""

    if true_matches:
        evidence.extend(
            evidence_from_matches(
                schema,
                CATEGORY,
                field,
                True,
                true_matches,
            )
        )
    if false_matches:
        evidence.extend(
            evidence_from_matches(
                schema,
                CATEGORY,
                field,
                False,
                false_matches,
            )
        )
    contradiction = bool(true_matches and false_matches)
    if true_matches and not false_matches:
        values[field] = True
    elif false_matches and not true_matches:
        values[field] = False
    return contradiction


def apply_flammable_rules(
    document: TextDocument,
    schema: EvidenceSchema,
    config: TextRuleConfig,
) -> RuleResult:
    """Extract conservative flammability evidence from name and description."""

    category_config = config.category(CATEGORY)
    matches = {
        name: find_signal_matches(
            document,
            signal,
            config.exclusion_window,
        )
        for name, signal in category_config.signals.items()
    }
    values: dict[str, object] = unknown_values(schema, CATEGORY)
    evidence: list = []
    contradictions = []

    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "fuel_present",
            matches["fuel_present"],
            matches["explicit_without_fuel"],
        )
    )
    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "fuel_included",
            matches["fuel_included"],
            matches["explicit_without_fuel"],
        )
    )
    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "requires_external_fuel",
            matches["requires_external_fuel"],
        )
    )
    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "independent_ignition_source",
            matches["independent_ignition_source"],
            matches["device_with_embedded_ignition"],
        )
    )
    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "flammable_item_in_kit",
            matches["flammable_item_in_kit"],
            matches["flammable_item_not_in_kit"],
        )
    )
    contradictions.append(
        _set_boolean(
            values,
            evidence,
            schema,
            "explicit_without_fuel",
            matches["explicit_without_fuel"],
        )
    )

    type_candidates = (
        ("kit", matches["flammable_item_in_kit"]),
        ("independent_ignition_source", matches["independent_ignition_source"]),
        ("combustible_content", matches["fuel_present"]),
        ("device_requiring_external_fuel", matches["requires_external_fuel"]),
        (
            "device_with_embedded_ignition",
            matches["device_with_embedded_ignition"],
        ),
        ("flammable_component", matches["flammable_component"]),
    )
    selected_type = next(
        ((object_type, found) for object_type, found in type_candidates if found),
        None,
    )
    if selected_type is not None:
        object_type, found = selected_type
        values["object_type"] = object_type
        evidence.extend(
            evidence_from_matches(
                schema,
                CATEGORY,
                "object_type",
                object_type,
                found,
            )
        )

    positive_rules = []
    if matches["independent_ignition_source"]:
        positive_rules.append("INDEPENDENT_IGNITION_SOURCE")
    if matches["fuel_present"]:
        positive_rules.append("FLAMMABLE_CONTENT")
    if matches["flammable_item_in_kit"]:
        positive_rules.append("FLAMMABLE_ITEM_IN_KIT")

    negative_rules = []
    if matches["requires_external_fuel"] and matches["explicit_without_fuel"]:
        negative_rules.append("DEVICE_WITHOUT_INCLUDED_FUEL")
    if matches["device_with_embedded_ignition"]:
        negative_rules.append("DEVICE_WITH_EMBEDDED_IGNITION")
    if matches["flammable_component"]:
        negative_rules.append("FLAMMABLE_COMPONENT_ONLY")
    if matches["explicit_without_fuel"] or matches["flammable_item_not_in_kit"]:
        negative_rules.append("NO_FLAMMABLE_CONTENT")

    conflict = bool(
        (positive_rules and negative_rules)
        or any(contradictions)
    )
    verdict: int | None = None
    rule_id = "INSUFFICIENT_EVIDENCE"
    if not conflict and positive_rules:
        verdict = 1
        rule_id = positive_rules[0]
    elif not conflict and negative_rules:
        verdict = 0
        rule_id = negative_rules[0]

    result = RuleResult(
        category=CATEGORY,
        values=values,
        evidence=tuple(evidence),
        rule_id=rule_id,
        verdict=verdict,
        has_negation=has_negation(document, config),
        conflict=conflict,
        matched_signals=tuple(name for name, found in matches.items() if found),
    )
    return validate_rule_result(result, schema)
