"""Text-only evidence rules for the dietary-supplement category."""

from __future__ import annotations

from ecup.rules.engine import (
    RuleResult,
    TextDocument,
    TextRuleConfig,
    evidence_from_matches,
    find_signal_matches,
    has_negation,
    unknown_values,
    validate_rule_result,
)
from ecup.rules.schema import EvidenceSchema

CATEGORY = "БАД"


def apply_bad_rules(
    document: TextDocument,
    schema: EvidenceSchema,
    config: TextRuleConfig,
) -> RuleResult:
    """Extract conservative BAD evidence without using label or images."""

    category_config = config.category(CATEGORY)
    matches = {
        name: find_signal_matches(
            document,
            signal,
            config.exclusion_window,
        )
        for name, signal in category_config.signals.items()
    }
    values = unknown_values(schema, CATEGORY)
    evidence = []

    for field in (
        "explicit_bad_marker",
        "dietary_supplement_marker",
        "explicit_not_bad",
        "sport_nutrition",
    ):
        if matches[field]:
            values[field] = True
            evidence.extend(
                evidence_from_matches(
                    schema,
                    CATEGORY,
                    field,
                    True,
                    matches[field],
                )
            )

    positive_matches = (
        *matches["explicit_bad_marker"],
        *matches["dietary_supplement_marker"],
    )
    if positive_matches:
        source = (
            "description"
            if any(match.source == "description" for match in positive_matches)
            else "name"
        )
        marker_match = next(
            match for match in positive_matches if match.source == source
        )
        values["marker_source"] = source
        evidence.extend(
            evidence_from_matches(
                schema,
                CATEGORY,
                "marker_source",
                source,
                (marker_match,),
            )
        )

    positive = bool(positive_matches)
    negative = bool(matches["explicit_not_bad"] or matches["sport_nutrition"])
    conflict = positive and negative
    verdict: int | None = None
    rule_id = "INSUFFICIENT_EVIDENCE"
    if not conflict and negative:
        verdict = 0
        rule_id = (
            "EXPLICIT_NOT_BAD"
            if matches["explicit_not_bad"]
            else "SPORT_NUTRITION"
        )
    elif not conflict and positive:
        verdict = 1
        rule_id = (
            "EXPLICIT_BAD_MARKER"
            if matches["explicit_bad_marker"]
            else "DIETARY_SUPPLEMENT_MARKER"
        )

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
