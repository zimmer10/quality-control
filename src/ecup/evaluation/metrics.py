"""Leak-free competition metric computed separately for both categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

BAD_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"
SUPPORTED_CATEGORIES = (BAD_CATEGORY, FLAMMABLE_CATEGORY)


@dataclass(frozen=True)
class CompetitionMetric:
    """F1 values for both competition categories and their mean."""

    f1_bad: float
    f1_flammable: float

    @property
    def mean_f1(self) -> float:
        """Return the competition score."""

        return (self.f1_bad + self.f1_flammable) / 2.0

    def as_dict(self) -> dict[str, float]:
        """Return stable names suitable for experiment reports."""

        return {
            "f1_bad": self.f1_bad,
            "f1_flammable": self.f1_flammable,
            "mean_f1": self.mean_f1,
        }


def binary_f1_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Match sklearn's binary F1 for 0/1 labels and zero_division=0."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    invalid_labels = (set(y_true) | set(y_pred)) - {0, 1}
    if invalid_labels:
        raise ValueError(f"labels must be binary 0/1, got: {invalid_labels}")

    true_positive = sum(
        true_label == 1 and predicted_label == 1
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )
    false_positive = sum(
        true_label == 0 and predicted_label == 1
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )
    false_negative = sum(
        true_label == 1 and predicted_label == 0
        for true_label, predicted_label in zip(y_true, y_pred, strict=True)
    )

    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return 2 * true_positive / denominator


def competition_f1_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
) -> CompetitionMetric:
    """Calculate category F1 values and their arithmetic mean."""

    if not (len(y_true) == len(y_pred) == len(categories)):
        raise ValueError("y_true, y_pred and categories must have equal lengths")

    unknown_categories = set(categories) - set(SUPPORTED_CATEGORIES)
    if unknown_categories:
        raise ValueError(f"unsupported categories: {unknown_categories}")

    scores: dict[str, float] = {}
    for category in SUPPORTED_CATEGORIES:
        indices = [index for index, value in enumerate(categories) if value == category]
        if not indices:
            raise ValueError(f"category is absent: {category}")
        scores[category] = binary_f1_score(
            [y_true[index] for index in indices],
            [y_pred[index] for index in indices],
        )

    return CompetitionMetric(
        f1_bad=scores[BAD_CATEGORY],
        f1_flammable=scores[FLAMMABLE_CATEGORY],
    )
