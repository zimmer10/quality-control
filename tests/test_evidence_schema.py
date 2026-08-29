from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from ecup.rules.schema import (
    EvidenceRecord,
    load_evidence_schema,
    main,
)


@pytest.fixture
def schema():
    return load_evidence_schema("configs/evidence.yaml")


def complete_bad_values(schema) -> dict[str, object]:
    values = {
        name: schema.unknown_value
        for name in schema.category("БАД").fields
    }
    values.update(
        explicit_bad_marker=True,
        marker_source="description",
    )
    return values


def load_yaml() -> dict[str, object]:
    with open("configs/evidence.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(tmp_path, payload: object):
    path = tmp_path / "evidence.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_canonical_schema_freezes_categories_fields_and_sources(schema) -> None:
    assert schema.version == 1
    assert schema.unknown_value == "unknown"
    assert tuple(schema.categories) == (
        "БАД",
        "Легковоспламеняющиеся",
    )
    assert tuple(schema.sources) == (
        "name",
        "description",
        "ocr",
        "image",
    )
    assert tuple(schema.category("БАД").fields) == (
        "explicit_bad_marker",
        "dietary_supplement_marker",
        "explicit_not_bad",
        "sport_nutrition",
        "marker_source",
    )
    assert tuple(schema.category("Легковоспламеняющиеся").fields) == (
        "object_type",
        "fuel_present",
        "fuel_included",
        "requires_external_fuel",
        "independent_ignition_source",
        "flammable_item_in_kit",
        "explicit_without_fuel",
    )


def test_text_evidence_is_normalized_and_serialized(schema) -> None:
    evidence = schema.parse_evidence(
        "БАД",
        {
            "field": "explicit_bad_marker",
            "value": True,
            "source": "description",
            "text": "  Биологически активная добавка  ",
        },
    )

    assert evidence == EvidenceRecord(
        field="explicit_bad_marker",
        value=True,
        source="description",
        text="Биологически активная добавка",
    )
    assert evidence.to_dict() == {
        "field": "explicit_bad_marker",
        "value": True,
        "source": "description",
        "text": "Биологически активная добавка",
    }


def test_ocr_and_image_sources_require_an_image_locator(schema) -> None:
    ocr = schema.parse_evidence(
        "БАД",
        {
            "field": "dietary_supplement_marker",
            "value": True,
            "source": "ocr",
            "text": "dietary supplement",
            "image_index": 2,
        },
    )
    image = schema.parse_evidence(
        "Легковоспламеняющиеся",
        {
            "field": "independent_ignition_source",
            "value": True,
            "source": "image",
            "image_index": 0,
        },
    )

    assert ocr.image_index == 2
    assert image.to_dict() == {
        "field": "independent_ignition_source",
        "value": True,
        "source": "image",
        "image_index": 0,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "field": "explicit_bad_marker",
                "value": True,
                "source": "description",
            },
            "requires evidence text",
        ),
        (
            {
                "field": "explicit_bad_marker",
                "value": True,
                "source": "ocr",
                "text": "БАД",
            },
            "requires image_index",
        ),
        (
            {
                "field": "explicit_bad_marker",
                "value": True,
                "source": "description",
                "text": "БАД",
                "image_index": 0,
            },
            "must not contain image_index",
        ),
        (
            {
                "field": "explicit_bad_marker",
                "value": "unknown",
                "source": "description",
                "text": "нет информации",
            },
            "not a confirmed evidence fact",
        ),
        (
            {
                "field": "missing_field",
                "value": True,
                "source": "description",
                "text": "БАД",
            },
            "unknown evidence field",
        ),
        (
            {
                "field": "explicit_bad_marker",
                "value": True,
                "source": "label",
                "text": "1",
            },
            "unknown evidence source",
        ),
        (
            {
                "field": "explicit_bad_marker",
                "value": 1,
                "source": "description",
                "text": "БАД",
            },
            "expects true, false or unknown",
        ),
    ],
)
def test_invalid_evidence_is_rejected(schema, payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        schema.parse_evidence("БАД", payload)


def test_evidence_list_is_strict(schema) -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        schema.parse_evidence_list("БАД", "not-a-list")

    with pytest.raises(ValueError, match="unknown fields"):
        schema.parse_evidence(
            "БАД",
            {
                "field": "explicit_bad_marker",
                "value": True,
                "source": "name",
                "text": "БАД",
                "label": 1,
            },
        )


def test_complete_lora_target_round_trip(schema) -> None:
    target = schema.build_target(
        "БАД",
        complete_bad_values(schema),
        "EXPLICIT_BAD_MARKER",
        1,
    )

    assert target["explicit_bad_marker"] is True
    assert target["dietary_supplement_marker"] == "unknown"
    assert target["marker_source"] == "description"
    assert target["rule_id"] == "EXPLICIT_BAD_MARKER"
    assert target["verdict"] == 1
    assert schema.validate_target("БАД", target) == target


def test_target_rejects_missing_extra_and_invalid_control_fields(schema) -> None:
    values = complete_bad_values(schema)

    with pytest.raises(ValueError, match="miss fields"):
        schema.build_target(
            "БАД",
            {"explicit_bad_marker": True},
            "EXPLICIT_BAD_MARKER",
            1,
        )
    with pytest.raises(ValueError, match="unknown evidence value fields: label"):
        schema.build_target(
            "БАД",
            {**values, "label": 1},
            "EXPLICIT_BAD_MARKER",
            1,
        )
    with pytest.raises(ValueError, match="unknown rule_id"):
        schema.build_target("БАД", values, "MADE_UP_RULE", 1)
    with pytest.raises(ValueError, match="integer 0 or 1"):
        schema.build_target("БАД", values, "EXPLICIT_BAD_MARKER", True)


def test_partial_values_are_supported_for_rule_engine_steps(schema) -> None:
    assert schema.validate_values(
        "Легковоспламеняющиеся",
        {"fuel_included": False},
        require_all=False,
    ) == {"fuel_included": False}

    with pytest.raises(ValueError, match="expects one of"):
        schema.validate_values(
            "Легковоспламеняющиеся",
            {"object_type": "lighter"},
            require_all=False,
        )


def test_loader_rejects_missing_category(tmp_path) -> None:
    payload = load_yaml()
    del payload["categories"]["БАД"]

    with pytest.raises(ValueError, match="categories must be exactly"):
        load_evidence_schema(write_yaml(tmp_path, payload))


def test_loader_rejects_changed_source_set(tmp_path) -> None:
    payload = load_yaml()
    del payload["sources"]["image"]

    with pytest.raises(ValueError, match="sources must be exactly"):
        load_evidence_schema(write_yaml(tmp_path, payload))


def test_loader_rejects_enum_without_unknown(tmp_path) -> None:
    payload = load_yaml()
    payload["categories"]["БАД"]["fields"]["marker_source"][
        "allowed_values"
    ] = ["name", "description"]

    with pytest.raises(ValueError, match="must contain unknown"):
        load_evidence_schema(write_yaml(tmp_path, payload))


def test_loader_rejects_duplicate_rule_ids(tmp_path) -> None:
    payload = load_yaml()
    payload["categories"]["БАД"]["rule_ids"].append(
        payload["categories"]["БАД"]["rule_ids"][0]
    )

    with pytest.raises(ValueError, match="contains duplicates"):
        load_evidence_schema(write_yaml(tmp_path, payload))


def test_loader_rejects_non_boolean_source_requirement(tmp_path) -> None:
    payload = deepcopy(load_yaml())
    payload["sources"]["ocr"]["requires_text"] = "yes"

    with pytest.raises(ValueError, match="requirements must be booleans"):
        load_evidence_schema(write_yaml(tmp_path, payload))


def test_cli_validates_schema_and_writes_report(tmp_path) -> None:
    report_path = tmp_path / "R08-evidence-schema.md"

    exit_code = main(
        [
            "--schema",
            "configs/evidence.yaml",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = report_path.read_text(encoding="utf-8")
    assert "# R08 — evidence schema" in report
    assert "## Категория: БАД" in report
    assert "## Категория: Легковоспламеняющиеся" in report
    assert '"rule_id": "DEVICE_WITHOUT_INCLUDED_FUEL"' in report
