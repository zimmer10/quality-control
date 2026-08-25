"""Tests for the competition metric."""

import unittest

from ecup.evaluation.metrics import binary_f1_score, competition_f1_score


class BinaryF1ScoreTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        self.assertEqual(binary_f1_score([0, 1, 1], [0, 1, 1]), 1.0)

    def test_zero_denominator(self) -> None:
        self.assertEqual(binary_f1_score([0, 0], [0, 0]), 0.0)

    def test_invalid_label(self) -> None:
        with self.assertRaises(ValueError):
            binary_f1_score([0, 2], [0, 1])


class CompetitionF1ScoreTests(unittest.TestCase):
    def test_category_scores_are_averaged(self) -> None:
        result = competition_f1_score(
            y_true=[1, 1, 0, 0, 1, 0],
            y_pred=[1, 0, 1, 0, 1, 0],
            categories=[
                "БАД",
                "БАД",
                "БАД",
                "БАД",
                "Легковоспламеняющиеся",
                "Легковоспламеняющиеся",
            ],
        )

        self.assertEqual(result.f1_bad, 0.5)
        self.assertEqual(result.f1_flammable, 1.0)
        self.assertEqual(result.mean_f1, 0.75)

    def test_missing_category_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            competition_f1_score([1], [1], ["БАД"])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            competition_f1_score([1, 0], [1], ["БАД", "БАД"])


if __name__ == "__main__":
    unittest.main()
