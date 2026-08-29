from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
import yaml

from ecup.features.text import build_text_rule_features
from ecup.models.fast import (
    ARTIFACT_FILENAMES,
    FAST_OOF_COLUMNS,
    FAST_OOF_SCHEMA,
    build_fold_splits,
    clean_text,
    compose_text,
    evaluate_fast_oof,
    fast_oof_checksum,
    fit_category_model,
    generate_fast_oof,
    load_category_model,
    load_fast_config,
    save_final_models,
    train_final_models,
    validate_fast_inputs,
)
from ecup.rules.engine import load_text_rule_config
from ecup.rules.schema import load_evidence_schema


def sample_source() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    identifier = 0
    for category in ("БАД", "Легковоспламеняющиеся"):
        for fold in range(3):
            for label in (0, 0, 1, 1):
                if category == "БАД":
                    name = (
                        "BCAA спортивный комплекс"
                        if label == 0
                        else "БАД витаминный комплекс"
                    )
                    description = (
                        "Спортивное питание и аминокислоты"
                        if label == 0
                        else "Биологически активная добавка к пище"
                    )
                else:
                    name = (
                        "Газовая плита туристическая"
                        if label == 0
                        else "Спички туристические"
                    )
                    description = (
                        "Работает от баллона, приобретаемого отдельно"
                        if label == 0
                        else "Самостоятельный источник открытого огня"
                    )
                rows.append(
                    {
                        "id": identifier,
                        "name": f"{name} {identifier}",
                        "description": description,
                        "category": category,
                        "label": label,
                        "_fold": fold,
                    }
                )
                identifier += 1
    return pl.DataFrame(rows)


def sample_folds(source: pl.DataFrame) -> pl.DataFrame:
    return source.select(
        "id",
        "category",
        "label",
        pl.concat_str(pl.lit("g"), pl.col("id")).alias("group_id"),
        pl.col("_fold").cast(pl.Int8).alias("fold"),
    )


def sample_rules(source: pl.DataFrame, folds: pl.DataFrame) -> pl.DataFrame:
    return build_text_rule_features(
        source,
        folds,
        load_evidence_schema("configs/evidence.yaml"),
        load_text_rule_config("configs/text_rules.yaml"),
    )


def small_config():
    config = load_fast_config("configs/fast_models.yaml")
    return replace(
        config,
        text=replace(
            config.text,
            word_min_df=1,
            char_min_df=1,
            word_max_features=1_000,
            char_max_features=2_000,
        ),
    )


def test_canonical_fast_config_is_versioned_and_category_specific() -> None:
    config = load_fast_config("configs/fast_models.yaml")

    assert config.version == 1
    assert config.seed == 42
    assert config.text.word_ngram_range == (1, 2)
    assert config.text.char_ngram_range == (3, 5)
    assert config.models.class_weight_by_category == {
        "БАД": None,
        "Легковоспламеняющиеся": "balanced",
    }
    assert config.paths.oof_path == Path("oof/fast.parquet")


def test_text_cleaning_removes_markup_and_keeps_field_boundary() -> None:
    assert clean_text("<p>БАД&nbsp; ТЕКСТ</p>") == "бад текст"
    assert compose_text("Название", None) == (
        "__name__ название __description__"
    )


def test_fit_model_uses_only_supplied_training_vocabulary() -> None:
    config = small_config()
    texts = [
        "negative sport food",
        "negative amino",
        "positive supplement",
        "positive vitamin",
    ]
    labels = [0, 0, 1, 1]

    model, summary = fit_category_model("БАД", texts, labels, config)
    predictions = model.predict(["validation_only_token supplement"])

    assert "validation_only_token" not in model.word_vectorizer.vocabulary_
    assert summary.rows == 4
    assert predictions.word_svc_margin.shape == (1,)
    assert predictions.char_svc_margin.shape == (1,)
    assert 0 <= predictions.text_logreg_probability[0] <= 1


def test_fold_splits_cover_each_row_once_without_group_leakage() -> None:
    source = sample_source()
    folds = sample_folds(source)
    splits = build_fold_splits(folds)

    assert len(splits) == 6
    validation_ids = [
        identifier
        for split in splits
        for identifier in split.validation_ids
    ]
    assert sorted(validation_ids) == sorted(source.get_column("id").to_list())
    assert all(
        not (split.train_groups & split.validation_groups)
        for split in splits
    )


def test_input_validation_rejects_misaligned_text_rules() -> None:
    source = sample_source()
    folds = sample_folds(source)
    rules = sample_rules(source, folds).with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit(1))
        .otherwise(pl.col("true_label"))
        .alias("true_label")
    )

    with pytest.raises(ValueError, match="text rules id/category/label"):
        validate_fast_inputs(source, folds, rules)


def test_oof_schema_alignment_metrics_and_reproducibility() -> None:
    source = sample_source()
    folds = sample_folds(source)
    rules = sample_rules(source, folds)
    config = small_config()

    first = generate_fast_oof(source, folds, rules, config)
    second = generate_fast_oof(source, folds, rules, config)

    assert first.frame.columns == list(FAST_OOF_COLUMNS)
    assert dict(first.frame.schema) == FAST_OOF_SCHEMA
    assert first.frame.height == source.height
    assert first.frame.null_count().row(0) == (0,) * len(FAST_OOF_COLUMNS)
    assert first.frame.equals(second.frame)
    assert fast_oof_checksum(first.frame) == fast_oof_checksum(second.frame)
    metrics = evaluate_fast_oof(first.frame)
    assert metrics["majority_vote"].mean_f1 == pytest.approx(1.0)
    assert set(first.frame.get_column("text_rule_verdict").to_list()) <= {
        -1,
        0,
        1,
    }


def test_final_artifacts_round_trip_without_prediction_drift(tmp_path) -> None:
    source = sample_source()
    folds = sample_folds(source)
    config = small_config()
    models, summaries, _ = train_final_models(source, folds, config)

    manifest = save_final_models(
        models,
        summaries,
        tmp_path,
        config,
        "test-oof-checksum",
    )
    artifact_path = tmp_path / ARTIFACT_FILENAMES["БАД"]
    payload = joblib.load(artifact_path)
    loaded = load_category_model(artifact_path)
    texts = ["биологически активная добавка", "спортивное питание"]
    expected = models["БАД"].predict(texts)
    actual = loaded.predict(texts)

    assert manifest["oof_checksum"] == "test-oof-checksum"
    assert payload["version"] == 1
    assert payload["category"] == "БАД"
    assert payload.__class__ is dict
    assert (tmp_path / "manifest.json").exists()
    np.testing.assert_allclose(
        actual.text_logreg_probability,
        expected.text_logreg_probability,
    )
    np.testing.assert_allclose(
        actual.word_svc_margin,
        expected.word_svc_margin,
    )


def test_config_loader_rejects_unknown_class_weight(tmp_path) -> None:
    with open("configs/fast_models.yaml", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    invalid = deepcopy(payload)
    invalid["models"]["class_weight_by_category"]["БАД"] = "automatic"
    path = tmp_path / "fast.yaml"
    path.write_text(
        yaml.safe_dump(invalid, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="null or balanced"):
        load_fast_config(path)


def test_cli_writes_oof_models_and_report(tmp_path) -> None:
    source = sample_source()
    folds = sample_folds(source)
    rules = sample_rules(source, folds)
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.parquet"
    rules_path = tmp_path / "rules.parquet"
    oof_path = tmp_path / "fast.parquet"
    artifacts_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.md"
    source.drop("_fold").write_csv(data_path)
    folds.write_parquet(folds_path)
    rules.write_parquet(rules_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ecup.models.fast",
            "--config",
            "configs/fast_models.yaml",
            "--data",
            str(data_path),
            "--folds",
            str(folds_path),
            "--text-rules",
            str(rules_path),
            "--oof",
            str(oof_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Fast OOF complete" in completed.stdout
    assert pl.read_parquet(oof_path).columns == list(FAST_OOF_COLUMNS)
    assert (artifacts_dir / "bad.joblib").exists()
    assert (artifacts_dir / "flammable.joblib").exists()
    assert (artifacts_dir / "manifest.json").exists()
    assert load_category_model(artifacts_dir / "bad.joblib").category == "БАД"
    report = report_path.read_text(encoding="utf-8")
    assert "# R10 — fast OOF" in report
    assert "Пересечений " in report
