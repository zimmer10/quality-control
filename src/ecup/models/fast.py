"""Fold-safe category-specific TF-IDF models and fast OOF generation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ecup.contracts import OOF_BASE_COLUMNS
from ecup.data.audit import load_dataset
from ecup.data.folds import FOLD_COLUMNS
from ecup.evaluation.metrics import (
    BAD_CATEGORY,
    FLAMMABLE_CATEGORY,
    SUPPORTED_CATEGORIES,
    CompetitionMetric,
    binary_f1_score,
    competition_f1_score,
)
from ecup.features.text import (
    BOOLEAN_SIGNAL_FEATURES,
    MARKER_SOURCE_FEATURES,
    OBJECT_TYPE_FEATURES,
    TEXT_RULE_COLUMNS,
    TEXT_RULE_SCHEMA,
    TRI_STATE_FEATURES,
)
from ecup.rules.engine import normalize_rule_text

DEFAULT_CONFIG_PATH = Path("configs/fast_models.yaml")
SOURCE_COLUMNS = ("id", "name", "description", "category", "label")
MODEL_OUTPUT_COLUMNS = (
    "word_svc_margin",
    "char_svc_margin",
    "text_logreg_probability",
    "text_models_disagree",
)
FAST_RULE_FEATURE_COLUMNS = (
    *TRI_STATE_FEATURES,
    *BOOLEAN_SIGNAL_FEATURES,
    *MARKER_SOURCE_FEATURES,
    *OBJECT_TYPE_FEATURES,
    "text_rule_verdict",
    "text_rule_has_verdict",
    "text_rule_conflict",
    "text_has_negation",
    "text_evidence_count",
    "text_name_evidence_count",
    "text_description_evidence_count",
)
FAST_OOF_COLUMNS = (
    *OOF_BASE_COLUMNS,
    *MODEL_OUTPUT_COLUMNS,
    *FAST_RULE_FEATURE_COLUMNS,
)
FAST_OOF_SCHEMA = {
    "id": pl.Int64,
    "category": pl.String,
    "fold": pl.Int8,
    "group_id": pl.String,
    "true_label": pl.Int64,
    "word_svc_margin": pl.Float32,
    "char_svc_margin": pl.Float32,
    "text_logreg_probability": pl.Float32,
    "text_models_disagree": pl.Boolean,
    **{
        column: TEXT_RULE_SCHEMA[column]
        for column in FAST_RULE_FEATURE_COLUMNS
    },
}
ARTIFACT_FILENAMES = {
    BAD_CATEGORY: "bad.joblib",
    FLAMMABLE_CATEGORY: "flammable.joblib",
}
MODEL_ARTIFACT_VERSION = 1
MODEL_ARTIFACT_KEYS = {
    "version",
    "category",
    "word_vectorizer",
    "char_vectorizer",
    "word_svc",
    "char_svc",
    "logistic_regression",
}
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class FastTextSettings:
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    word_min_df: int
    char_min_df: int
    word_max_features: int
    char_max_features: int
    sublinear_tf: bool


@dataclass(frozen=True)
class FastModelSettings:
    linear_svc_c: float
    logistic_regression_c: float
    logistic_max_iter: int
    class_weight_by_category: Mapping[str, str | None]


@dataclass(frozen=True)
class FastPaths:
    data_path: Path
    folds_path: Path
    text_rules_path: Path
    oof_path: Path
    artifacts_dir: Path
    report_path: Path


@dataclass(frozen=True)
class FastConfig:
    version: int
    seed: int
    text: FastTextSettings
    models: FastModelSettings
    paths: FastPaths


@dataclass(frozen=True)
class FastPredictions:
    word_svc_margin: np.ndarray
    char_svc_margin: np.ndarray
    text_logreg_probability: np.ndarray
    text_models_disagree: np.ndarray


@dataclass(frozen=True)
class CategoryFitSummary:
    category: str
    rows: int
    word_features: int
    char_features: int


@dataclass
class CategoryFastModel:
    """Serializable final model bundle for one known category."""

    category: str
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    word_svc: LinearSVC
    char_svc: LinearSVC
    logistic_regression: LogisticRegression

    def predict(self, texts: Sequence[str]) -> FastPredictions:
        text_values = list(texts)
        word_matrix = self.word_vectorizer.transform(text_values)
        char_matrix = self.char_vectorizer.transform(text_values)
        combined = hstack(
            [word_matrix, char_matrix],
            format="csr",
            dtype=np.float32,
        )
        word_margin = np.asarray(
            self.word_svc.decision_function(word_matrix),
            dtype=np.float32,
        )
        char_margin = np.asarray(
            self.char_svc.decision_function(char_matrix),
            dtype=np.float32,
        )
        positive_index = int(
            np.flatnonzero(self.logistic_regression.classes_ == 1)[0]
        )
        probability = np.asarray(
            self.logistic_regression.predict_proba(combined)[:, positive_index],
            dtype=np.float32,
        )
        word_label = word_margin >= 0
        char_label = char_margin >= 0
        logreg_label = probability >= 0.5
        disagree = ~(
            (word_label == char_label)
            & (char_label == logreg_label)
        )
        return FastPredictions(
            word_svc_margin=word_margin,
            char_svc_margin=char_margin,
            text_logreg_probability=probability,
            text_models_disagree=np.asarray(disagree, dtype=np.bool_),
        )


@dataclass(frozen=True)
class FoldSplit:
    category: str
    fold: int
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    train_groups: frozenset[str]
    validation_groups: frozenset[str]


@dataclass(frozen=True)
class FoldRunSummary:
    category: str
    fold: int
    train_rows: int
    validation_rows: int
    word_features: int
    char_features: int
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True)
class OOFRun:
    frame: pl.DataFrame
    folds: tuple[FoldRunSummary, ...]
    elapsed_seconds: float


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


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _positive_float(value: object, path: str) -> float:
    if type(value) not in (int, float) or float(value) <= 0:
        raise ValueError(f"{path} must be a positive number")
    return float(value)


def _ngram_range(value: object, path: str) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{path} must contain two integers")
    lower, upper = value
    if (
        type(lower) is not int
        or type(upper) is not int
        or lower <= 0
        or lower > upper
    ):
        raise ValueError(f"{path} must be an ordered positive integer pair")
    return lower, upper


def load_fast_config(path: str | Path = DEFAULT_CONFIG_PATH) -> FastConfig:
    """Load and strictly validate the versioned R10 configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _as_mapping(payload, "fast_models")
    _exact_keys(
        root,
        {"version", "seed", "input", "validation", "text", "models", "output"},
        "fast_models",
    )
    version = root["version"]
    seed = root["seed"]
    if type(version) is not int or version != 1:
        raise ValueError("fast_models.version must be integer 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("fast_models.seed must be a non-negative integer")

    raw_input = _as_mapping(root["input"], "fast_models.input")
    _exact_keys(raw_input, {"data_path", "text_rules_path"}, "fast_models.input")
    raw_validation = _as_mapping(
        root["validation"],
        "fast_models.validation",
    )
    _exact_keys(raw_validation, {"folds_path"}, "fast_models.validation")
    raw_output = _as_mapping(root["output"], "fast_models.output")
    _exact_keys(
        raw_output,
        {"oof_path", "artifacts_dir", "report_path"},
        "fast_models.output",
    )
    for mapping, keys, prefix in (
        (raw_input, ("data_path", "text_rules_path"), "fast_models.input"),
        (raw_validation, ("folds_path",), "fast_models.validation"),
        (
            raw_output,
            ("oof_path", "artifacts_dir", "report_path"),
            "fast_models.output",
        ),
    ):
        for key in keys:
            if not isinstance(mapping[key], str) or not mapping[key]:
                raise ValueError(f"{prefix}.{key} must be a non-empty path")

    raw_text = _as_mapping(root["text"], "fast_models.text")
    _exact_keys(
        raw_text,
        {
            "word_ngram_range",
            "char_ngram_range",
            "word_min_df",
            "char_min_df",
            "word_max_features",
            "char_max_features",
            "sublinear_tf",
        },
        "fast_models.text",
    )
    sublinear_tf = raw_text["sublinear_tf"]
    if type(sublinear_tf) is not bool:
        raise ValueError("fast_models.text.sublinear_tf must be boolean")
    text_settings = FastTextSettings(
        word_ngram_range=_ngram_range(
            raw_text["word_ngram_range"],
            "fast_models.text.word_ngram_range",
        ),
        char_ngram_range=_ngram_range(
            raw_text["char_ngram_range"],
            "fast_models.text.char_ngram_range",
        ),
        word_min_df=_positive_int(
            raw_text["word_min_df"],
            "fast_models.text.word_min_df",
        ),
        char_min_df=_positive_int(
            raw_text["char_min_df"],
            "fast_models.text.char_min_df",
        ),
        word_max_features=_positive_int(
            raw_text["word_max_features"],
            "fast_models.text.word_max_features",
        ),
        char_max_features=_positive_int(
            raw_text["char_max_features"],
            "fast_models.text.char_max_features",
        ),
        sublinear_tf=sublinear_tf,
    )

    raw_models = _as_mapping(root["models"], "fast_models.models")
    _exact_keys(
        raw_models,
        {
            "linear_svc_c",
            "logistic_regression_c",
            "logistic_max_iter",
            "class_weight_by_category",
        },
        "fast_models.models",
    )
    raw_weights = _as_mapping(
        raw_models["class_weight_by_category"],
        "fast_models.models.class_weight_by_category",
    )
    if set(raw_weights) != set(SUPPORTED_CATEGORIES):
        raise ValueError(
            "class_weight_by_category must contain exactly: "
            + ", ".join(SUPPORTED_CATEGORIES)
        )
    weights: dict[str, str | None] = {}
    for category in SUPPORTED_CATEGORIES:
        value = raw_weights[category]
        if value not in (None, "balanced"):
            raise ValueError(
                f"class weight for {category} must be null or balanced"
            )
        weights[category] = value
    model_settings = FastModelSettings(
        linear_svc_c=_positive_float(
            raw_models["linear_svc_c"],
            "fast_models.models.linear_svc_c",
        ),
        logistic_regression_c=_positive_float(
            raw_models["logistic_regression_c"],
            "fast_models.models.logistic_regression_c",
        ),
        logistic_max_iter=_positive_int(
            raw_models["logistic_max_iter"],
            "fast_models.models.logistic_max_iter",
        ),
        class_weight_by_category=weights,
    )
    return FastConfig(
        version=version,
        seed=seed,
        text=text_settings,
        models=model_settings,
        paths=FastPaths(
            data_path=Path(raw_input["data_path"]),
            folds_path=Path(raw_validation["folds_path"]),
            text_rules_path=Path(raw_input["text_rules_path"]),
            oof_path=Path(raw_output["oof_path"]),
            artifacts_dir=Path(raw_output["artifacts_dir"]),
            report_path=Path(raw_output["report_path"]),
        ),
    )


def clean_text(value: object) -> str:
    """Remove markup and normalize one text field deterministically."""

    if value is None:
        return ""
    without_markup = HTML_TAG_PATTERN.sub(" ", html.unescape(str(value)))
    return normalize_rule_text(without_markup)


def compose_text(name: object, description: object) -> str:
    """Keep field boundaries while exposing only name and description."""

    return (
        "__name__ "
        + clean_text(name)
        + " __description__ "
        + clean_text(description)
    ).strip()


def compose_texts(frame: pl.DataFrame) -> list[str]:
    return [
        compose_text(name, description)
        for name, description in frame.select(
            "name",
            "description",
        ).iter_rows()
    ]


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


def _validate_unique_ids(frame: pl.DataFrame, table_name: str) -> None:
    if frame.get_column("id").null_count():
        raise ValueError(f"{table_name} id contains null values")
    if frame.get_column("id").n_unique() != frame.height:
        raise ValueError(f"{table_name} id values must be unique")


def validate_fast_inputs(
    source: pl.DataFrame,
    folds: pl.DataFrame,
    text_rules: pl.DataFrame,
) -> None:
    """Validate the three shared R10 inputs before any supervised fitting."""

    _require_columns(source, SOURCE_COLUMNS, "source")
    if folds.columns != list(FOLD_COLUMNS):
        raise ValueError(f"fold columns must be exactly: {FOLD_COLUMNS}")
    if text_rules.columns != list(TEXT_RULE_COLUMNS):
        raise ValueError("text rule columns do not match the frozen R09 contract")
    if dict(text_rules.schema) != TEXT_RULE_SCHEMA:
        raise ValueError("text rule dtypes do not match the frozen R09 contract")
    if source.is_empty():
        raise ValueError("source dataset is empty")
    for frame, table_name in (
        (source, "source"),
        (folds, "folds"),
        (text_rules, "text rules"),
    ):
        _validate_unique_ids(frame, table_name)
    if not (source.height == folds.height == text_rules.height):
        raise ValueError("source, folds and text rules row counts differ")

    source_contract = source.select("id", "category", "label").sort("id")
    fold_contract = folds.select("id", "category", "label").sort("id")
    if not source_contract.equals(fold_contract):
        raise ValueError("source and folds id/category/label values differ")
    rule_contract = text_rules.select(
        "id",
        "category",
        pl.col("true_label").alias("label"),
    ).sort("id")
    if not source_contract.equals(rule_contract):
        raise ValueError("source and text rules id/category/label values differ")
    expected_rule_base = folds.select(
        "id",
        "category",
        "fold",
        "group_id",
        pl.col("label").alias("true_label"),
    ).sort("id")
    if not expected_rule_base.equals(
        text_rules.select(OOF_BASE_COLUMNS).sort("id")
    ):
        raise ValueError("text rule base columns differ from folds")

    categories = set(source.get_column("category").to_list())
    if categories != set(SUPPORTED_CATEGORIES):
        raise ValueError(
            "source categories must be exactly: "
            + ", ".join(SUPPORTED_CATEGORIES)
        )
    invalid_labels = set(folds.get_column("label").to_list()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"fold labels must be binary: {invalid_labels}")
    fold_values = sorted(set(folds.get_column("fold").to_list()))
    if len(fold_values) < 2 or fold_values != list(range(len(fold_values))):
        raise ValueError("fold values must be contiguous from zero")
    group_leaks = (
        folds.group_by("group_id")
        .agg(pl.col("fold").n_unique().alias("fold_count"))
        .filter(pl.col("fold_count") != 1)
    )
    if not group_leaks.is_empty():
        raise ValueError("a group_id is assigned to multiple folds")

    for category in SUPPORTED_CATEGORIES:
        category_rows = folds.filter(pl.col("category") == category)
        if category_rows.is_empty():
            raise ValueError(f"category is absent: {category}")
        for fold in fold_values:
            train_labels = set(
                category_rows.filter(pl.col("fold") != fold)
                .get_column("label")
                .to_list()
            )
            if train_labels != {0, 1}:
                raise ValueError(
                    f"training split {category}, fold {fold} misses a class"
                )


def build_fold_splits(folds: pl.DataFrame) -> tuple[FoldSplit, ...]:
    """Build explicit category/fold splits and prove group disjointness."""

    fold_values = sorted(set(folds.get_column("fold").to_list()))
    splits: list[FoldSplit] = []
    for category in SUPPORTED_CATEGORIES:
        category_rows = folds.filter(pl.col("category") == category)
        for fold in fold_values:
            train = category_rows.filter(pl.col("fold") != fold)
            validation = category_rows.filter(pl.col("fold") == fold)
            if train.is_empty() or validation.is_empty():
                raise ValueError(f"empty split for {category}, fold {fold}")
            train_groups = frozenset(
                map(str, train.get_column("group_id").to_list())
            )
            validation_groups = frozenset(
                map(str, validation.get_column("group_id").to_list())
            )
            if train_groups & validation_groups:
                raise ValueError(
                    f"group leakage in {category}, fold {fold}"
                )
            splits.append(
                FoldSplit(
                    category=category,
                    fold=int(fold),
                    train_ids=tuple(map(int, train.get_column("id").to_list())),
                    validation_ids=tuple(
                        map(int, validation.get_column("id").to_list())
                    ),
                    train_groups=train_groups,
                    validation_groups=validation_groups,
                )
            )
    return tuple(splits)


def _vectorizers(
    settings: FastTextSettings,
) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    common = {
        "lowercase": False,
        "sublinear_tf": settings.sublinear_tf,
        "dtype": np.float32,
    }
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=settings.word_ngram_range,
        min_df=settings.word_min_df,
        max_features=settings.word_max_features,
        **common,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=settings.char_ngram_range,
        min_df=settings.char_min_df,
        max_features=settings.char_max_features,
        **common,
    )
    return word, char


def fit_category_model(
    category: str,
    texts: Sequence[str],
    labels: Sequence[int],
    config: FastConfig,
) -> tuple[CategoryFastModel, CategoryFitSummary]:
    """Fit fresh vectorizers and classifiers for exactly one train split."""

    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and aligned")
    if set(labels) != {0, 1}:
        raise ValueError("category training labels must contain both classes")
    word_vectorizer, char_vectorizer = _vectorizers(config.text)
    word_matrix = word_vectorizer.fit_transform(list(texts))
    char_matrix = char_vectorizer.fit_transform(list(texts))
    combined = hstack(
        [word_matrix, char_matrix],
        format="csr",
        dtype=np.float32,
    )
    class_weight = config.models.class_weight_by_category[category]
    word_svc = LinearSVC(
        C=config.models.linear_svc_c,
        class_weight=class_weight,
        dual="auto",
        random_state=config.seed,
    ).fit(word_matrix, labels)
    char_svc = LinearSVC(
        C=config.models.linear_svc_c,
        class_weight=class_weight,
        dual="auto",
        random_state=config.seed,
    ).fit(char_matrix, labels)
    logistic_regression = LogisticRegression(
        C=config.models.logistic_regression_c,
        class_weight=class_weight,
        max_iter=config.models.logistic_max_iter,
        random_state=config.seed,
        solver="liblinear",
    ).fit(combined, labels)
    model = CategoryFastModel(
        category=category,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        word_svc=word_svc,
        char_svc=char_svc,
        logistic_regression=logistic_regression,
    )
    summary = CategoryFitSummary(
        category=category,
        rows=len(texts),
        word_features=word_matrix.shape[1],
        char_features=char_matrix.shape[1],
    )
    return model, summary


def _training_table(source: pl.DataFrame, folds: pl.DataFrame) -> pl.DataFrame:
    return (
        source.select("id", "name", "description")
        .join(
            folds.select(FOLD_COLUMNS),
            on="id",
            how="inner",
            validate="1:1",
        )
        .sort("id")
    )


def _prediction_frame(
    ids: Sequence[int],
    predictions: FastPredictions,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ids,
            "word_svc_margin": predictions.word_svc_margin,
            "char_svc_margin": predictions.char_svc_margin,
            "text_logreg_probability": predictions.text_logreg_probability,
            "text_models_disagree": predictions.text_models_disagree,
        }
    ).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("word_svc_margin").cast(pl.Float32),
        pl.col("char_svc_margin").cast(pl.Float32),
        pl.col("text_logreg_probability").cast(pl.Float32),
        pl.col("text_models_disagree").cast(pl.Boolean),
    )


def _model_ready_rule_features(text_rules: pl.DataFrame) -> pl.DataFrame:
    return text_rules.select("id", *FAST_RULE_FEATURE_COLUMNS).with_columns(
        pl.col("text_rule_verdict").fill_null(-1).cast(pl.Int8)
    )


def generate_fast_oof(
    source: pl.DataFrame,
    folds: pl.DataFrame,
    text_rules: pl.DataFrame,
    config: FastConfig,
) -> OOFRun:
    """Generate one prediction per row with fold-fitted vectorizers/models."""

    validate_fast_inputs(source, folds, text_rules)
    table = _training_table(source, folds)
    prediction_frames: list[pl.DataFrame] = []
    run_summaries: list[FoldRunSummary] = []
    started = time.perf_counter()
    for split in build_fold_splits(folds):
        train = table.filter(pl.col("id").is_in(split.train_ids)).sort("id")
        validation = table.filter(
            pl.col("id").is_in(split.validation_ids)
        ).sort("id")
        fit_started = time.perf_counter()
        model, fit_summary = fit_category_model(
            split.category,
            compose_texts(train),
            train.get_column("label").to_list(),
            config,
        )
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = model.predict(compose_texts(validation))
        predict_seconds = time.perf_counter() - predict_started
        prediction_frames.append(
            _prediction_frame(
                validation.get_column("id").to_list(),
                predictions,
            )
        )
        run_summaries.append(
            FoldRunSummary(
                category=split.category,
                fold=split.fold,
                train_rows=train.height,
                validation_rows=validation.height,
                word_features=fit_summary.word_features,
                char_features=fit_summary.char_features,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )
        )

    model_predictions = pl.concat(prediction_frames).sort("id")
    base = folds.select(
        "id",
        "category",
        "fold",
        "group_id",
        pl.col("label").alias("true_label"),
    )
    oof = (
        base.join(
            model_predictions,
            on="id",
            how="inner",
            validate="1:1",
        )
        .join(
            _model_ready_rule_features(text_rules),
            on="id",
            how="inner",
            validate="1:1",
        )
        .select(FAST_OOF_COLUMNS)
        .sort("id")
    )
    validate_fast_oof(folds, oof)
    return OOFRun(
        frame=oof,
        folds=tuple(run_summaries),
        elapsed_seconds=time.perf_counter() - started,
    )


def validate_fast_oof(folds: pl.DataFrame, oof: pl.DataFrame) -> None:
    """Enforce the frozen fast OOF schema and row alignment."""

    if oof.columns != list(FAST_OOF_COLUMNS):
        raise ValueError(f"fast OOF columns must be exactly: {FAST_OOF_COLUMNS}")
    if dict(oof.schema) != FAST_OOF_SCHEMA:
        raise ValueError("fast OOF dtypes do not match the frozen schema")
    if oof.height != folds.height:
        raise ValueError("fast OOF must contain exactly one row per train row")
    _validate_unique_ids(oof, "fast OOF")
    expected_base = folds.select(
        "id",
        "category",
        "fold",
        "group_id",
        pl.col("label").alias("true_label"),
    ).sort("id")
    if not expected_base.equals(oof.select(OOF_BASE_COLUMNS).sort("id")):
        raise ValueError("fast OOF base columns differ from folds")
    if sum(oof.null_count().row(0)) != 0:
        raise ValueError("fast OOF contains null values")
    invalid_probability = oof.filter(
        ~pl.col("text_logreg_probability").is_between(0.0, 1.0)
    )
    if not invalid_probability.is_empty():
        raise ValueError("logistic probabilities must be within [0, 1]")
    expected_disagreement = ~(
        ((pl.col("word_svc_margin") >= 0) == (pl.col("char_svc_margin") >= 0))
        & (
            (pl.col("char_svc_margin") >= 0)
            == (pl.col("text_logreg_probability") >= 0.5)
        )
    )
    if oof.filter(
        pl.col("text_models_disagree") != expected_disagreement
    ).height:
        raise ValueError("text_models_disagree is inconsistent")


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fast_oof_checksum(oof: pl.DataFrame) -> str:
    """Hash logical OOF rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in oof.select(FAST_OOF_COLUMNS).sort("id").iter_rows():
        digest.update(_stable_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_fast_oof(oof: pl.DataFrame) -> dict[str, CompetitionMetric]:
    """Evaluate fixed-threshold component predictions and majority vote."""

    true_labels = oof.get_column("true_label").to_list()
    categories = oof.get_column("category").to_list()
    word = (
        oof.get_column("word_svc_margin") >= 0
    ).cast(pl.Int8).to_list()
    char = (
        oof.get_column("char_svc_margin") >= 0
    ).cast(pl.Int8).to_list()
    logreg = (
        oof.get_column("text_logreg_probability") >= 0.5
    ).cast(pl.Int8).to_list()
    majority = [
        int(word_value + char_value + logreg_value >= 2)
        for word_value, char_value, logreg_value in zip(
            word,
            char,
            logreg,
            strict=True,
        )
    ]
    return {
        "word_svc": competition_f1_score(true_labels, word, categories),
        "char_svc": competition_f1_score(true_labels, char, categories),
        "logistic_regression": competition_f1_score(
            true_labels,
            logreg,
            categories,
        ),
        "majority_vote": competition_f1_score(
            true_labels,
            majority,
            categories,
        ),
    }


def train_final_models(
    source: pl.DataFrame,
    folds: pl.DataFrame,
    config: FastConfig,
) -> tuple[
    dict[str, CategoryFastModel],
    dict[str, CategoryFitSummary],
    float,
]:
    """Fit one inference model per category on all available train rows."""

    table = _training_table(source, folds)
    models: dict[str, CategoryFastModel] = {}
    summaries: dict[str, CategoryFitSummary] = {}
    started = time.perf_counter()
    for category in SUPPORTED_CATEGORIES:
        category_rows = table.filter(
            pl.col("category") == category
        ).sort("id")
        model, summary = fit_category_model(
            category,
            compose_texts(category_rows),
            category_rows.get_column("label").to_list(),
            config,
        )
        models[category] = model
        summaries[category] = summary
    return models, summaries, time.perf_counter() - started


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_final_models(
    models: Mapping[str, CategoryFastModel],
    summaries: Mapping[str, CategoryFitSummary],
    artifacts_dir: str | Path,
    config: FastConfig,
    oof_checksum: str,
) -> dict[str, object]:
    """Persist final inference bundles and a reviewable artifact manifest."""

    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_entries: dict[str, object] = {}
    for category in SUPPORTED_CATEGORIES:
        model = models[category]
        filename = ARTIFACT_FILENAMES[category]
        path = directory / filename
        payload = {
            "version": MODEL_ARTIFACT_VERSION,
            "category": model.category,
            "word_vectorizer": model.word_vectorizer,
            "char_vectorizer": model.char_vectorizer,
            "word_svc": model.word_svc,
            "char_svc": model.char_svc,
            "logistic_regression": model.logistic_regression,
        }
        joblib.dump(payload, path, compress=3)
        summary = summaries[category]
        model_entries[category] = {
            "file": filename,
            "sha256": _file_checksum(path),
            "rows": summary.rows,
            "word_features": summary.word_features,
            "char_features": summary.char_features,
        }
    manifest: dict[str, object] = {
        "version": 1,
        "config_version": config.version,
        "seed": config.seed,
        "oof_checksum": oof_checksum,
        "model_output_columns": list(MODEL_OUTPUT_COLUMNS),
        "rule_feature_columns": list(FAST_RULE_FEATURE_COLUMNS),
        "models": model_entries,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_category_model(path: str | Path) -> CategoryFastModel:
    """Load a model bundle without depending on the CLI module name."""

    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping) or set(payload) != MODEL_ARTIFACT_KEYS:
        raise ValueError("model artifact has an invalid schema")
    if payload["version"] != MODEL_ARTIFACT_VERSION:
        raise ValueError("model artifact has an unsupported version")
    category = payload["category"]
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError("artifact contains an unsupported category")
    components = {
        "word_vectorizer": (payload["word_vectorizer"], TfidfVectorizer),
        "char_vectorizer": (payload["char_vectorizer"], TfidfVectorizer),
        "word_svc": (payload["word_svc"], LinearSVC),
        "char_svc": (payload["char_svc"], LinearSVC),
        "logistic_regression": (
            payload["logistic_regression"],
            LogisticRegression,
        ),
    }
    for name, (component, expected_type) in components.items():
        if not isinstance(component, expected_type):
            raise ValueError(f"model artifact component {name} is invalid")
    return CategoryFastModel(
        category=category,
        word_vectorizer=payload["word_vectorizer"],
        char_vectorizer=payload["char_vectorizer"],
        word_svc=payload["word_svc"],
        char_svc=payload["char_svc"],
        logistic_regression=payload["logistic_regression"],
    )


def measure_final_inference(
    models: Mapping[str, CategoryFastModel],
    source: pl.DataFrame,
) -> tuple[int, float]:
    started = time.perf_counter()
    rows = 0
    for category in SUPPORTED_CATEGORIES:
        category_rows = source.filter(
            pl.col("category") == category
        ).sort("id")
        models[category].predict(compose_texts(category_rows))
        rows += category_rows.height
    return rows, time.perf_counter() - started


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _majority_predictions(frame: pl.DataFrame) -> list[int]:
    word = (frame.get_column("word_svc_margin") >= 0).cast(pl.Int8)
    char = (frame.get_column("char_svc_margin") >= 0).cast(pl.Int8)
    logreg = (
        frame.get_column("text_logreg_probability") >= 0.5
    ).cast(pl.Int8)
    return ((word + char + logreg) >= 2).cast(pl.Int8).to_list()


def render_fast_report(
    oof_run: OOFRun,
    config: FastConfig,
    config_path: str | Path,
    oof_path: str | Path,
    artifacts_dir: str | Path,
    metrics: Mapping[str, CompetitionMetric],
    final_summaries: Mapping[str, CategoryFitSummary],
    final_fit_seconds: float,
    inference_rows: int,
    inference_seconds: float,
) -> str:
    """Render metrics, leak checks, dimensions and timing for R10."""

    oof = oof_run.frame
    lines = [
        "# R10 — fast OOF",
        "",
        "## Контракт",
        "",
        f"- Конфигурация: `{config_path}`, версия **{config.version}**.",
        f"- OOF-артефакт: `{oof_path}`.",
        f"- Финальные модели: `{artifacts_dir}`.",
        f"- Строк: **{oof.height}**.",
        f"- Колонок: **{len(oof.columns)}**.",
        f"- Групп: **{oof.get_column('group_id').n_unique()}**.",
        f"- Логическая SHA-256: `{fast_oof_checksum(oof)}`.",
        "- Одна строка соответствует одному train-товару.",
        "- TF-IDF и классификаторы обучались заново внутри каждого train-fold.",
        "- Пересечений `group_id` между train и validation нет.",
        "- OCR и изображения не использовались.",
        "- Числовые Text Rule features R09 присоединены без JSON/evidence-полей.",
        "",
        "## Конфигурация моделей",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        (
            f"| Word n-grams | {config.text.word_ngram_range[0]}–"
            f"{config.text.word_ngram_range[1]} |"
        ),
        (
            f"| Character n-grams | {config.text.char_ngram_range[0]}–"
            f"{config.text.char_ngram_range[1]} |"
        ),
        f"| LinearSVC C | {config.models.linear_svc_c} |",
        f"| LogisticRegression C | {config.models.logistic_regression_c} |",
        (
            "| Class weight БАД | "
            f"{config.models.class_weight_by_category[BAD_CATEGORY]} |"
        ),
        (
            "| Class weight Легковоспламеняющиеся | "
            f"{config.models.class_weight_by_category[FLAMMABLE_CATEGORY]} |"
        ),
        "",
        "## OOF-метрики при стандартных порогах",
        "",
        "| Компонент | F1 БАД | F1 Легковоспламеняющиеся | mean F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        lines.append(
            f"| `{name}` | {metric.f1_bad:.6f} | "
            f"{metric.f1_flammable:.6f} | {metric.mean_f1:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Диагностика folds",
            "",
            "| Категория | Fold | Train | Validation | Word features | Char features | Fit, s | Predict, s | Majority F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in oof_run.folds:
        fold_rows = oof.filter(
            (pl.col("category") == summary.category)
            & (pl.col("fold") == summary.fold)
        )
        fold_f1 = binary_f1_score(
            fold_rows.get_column("true_label").to_list(),
            _majority_predictions(fold_rows),
        )
        lines.append(
            f"| {summary.category} | {summary.fold} | "
            f"{summary.train_rows} | {summary.validation_rows} | "
            f"{summary.word_features} | {summary.char_features} | "
            f"{summary.fit_seconds:.3f} | {summary.predict_seconds:.3f} | "
            f"{fold_f1:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Финальные модели",
            "",
            "| Категория | Train rows | Word features | Char features |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in SUPPORTED_CATEGORIES:
        summary = final_summaries[category]
        lines.append(
            f"| {category} | {summary.rows} | {summary.word_features} | "
            f"{summary.char_features} |"
        )

    disagreement = oof.group_by("category").agg(
        pl.col("text_models_disagree").mean().alias("share")
    ).sort("category")
    lines.extend(
        [
            "",
            "## Время",
            "",
            f"- OOF-обучение и предсказание: **{oof_run.elapsed_seconds:.3f} s**.",
            f"- Финальное обучение на всём train: **{final_fit_seconds:.3f} s**.",
            (
                f"- Финальный text inference: **{inference_seconds:.3f} s** "
                f"для {inference_rows} строк "
                f"({1000 * inference_seconds / inference_rows:.3f} ms/row)."
            ),
            "",
            "## Расхождение моделей",
            "",
            "| Категория | Доля строк с расхождением |",
            "|---|---:|",
        ]
    )
    for row in disagreement.iter_rows(named=True):
        lines.append(f"| {row['category']} | {_percent(row['share'])} |")
    lines.extend(
        [
            "",
            "Пороги в R10 не подбирались: SVC использует margin 0, Logistic Regression — 0.5. Сырые OOF-сигналы предназначены для Router и meta-model на следующих этапах.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fold-safe fast text models and build OOF predictions"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--text-rules", type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_fast_config(args.config)
    data_path = args.data or config.paths.data_path
    folds_path = args.folds or config.paths.folds_path
    text_rules_path = args.text_rules or config.paths.text_rules_path
    oof_path = args.oof or config.paths.oof_path
    artifacts_dir = args.artifacts_dir or config.paths.artifacts_dir
    report_path = args.report or config.paths.report_path

    source = load_dataset(data_path)
    folds = pl.read_parquet(folds_path)
    text_rules = pl.read_parquet(text_rules_path)
    oof_run = generate_fast_oof(source, folds, text_rules, config)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_run.frame.write_parquet(oof_path, compression="zstd")
    checksum = fast_oof_checksum(oof_run.frame)
    metrics = evaluate_fast_oof(oof_run.frame)

    final_models, final_summaries, final_fit_seconds = train_final_models(
        source,
        folds,
        config,
    )
    save_final_models(
        final_models,
        final_summaries,
        artifacts_dir,
        config,
        checksum,
    )
    inference_rows, inference_seconds = measure_final_inference(
        final_models,
        source,
    )
    report = render_fast_report(
        oof_run,
        config,
        args.config,
        oof_path,
        artifacts_dir,
        metrics,
        final_summaries,
        final_fit_seconds,
        inference_rows,
        inference_seconds,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    majority = metrics["majority_vote"]
    print(
        f"Fast OOF complete: {oof_run.frame.height} rows, "
        f"mean_f1={majority.mean_f1:.6f}, checksum={checksum}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
