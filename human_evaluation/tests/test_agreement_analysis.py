import unittest

from scripts.analyze_agreement import (
    BINARY_LABELS,
    THREE_LABELS,
    classification_metrics,
    consensus,
    fleiss_kappa,
    score_to_three_class,
    three_to_binary,
)


class AgreementAnalysisTest(unittest.TestCase):
    def test_score_mapping_collapses_two_and_three(self):
        self.assertEqual(score_to_three_class(0), "easy_negative")
        self.assertEqual(score_to_three_class(1), "hard_negative")
        self.assertEqual(score_to_three_class(2), "positive")
        self.assertEqual(score_to_three_class(3), "positive")
        self.assertEqual(three_to_binary("easy_negative"), "negative")
        self.assertEqual(three_to_binary("hard_negative"), "negative")

    def test_three_way_tie_uses_flagged_ordinal_median(self):
        result = consensus(["positive", "easy_negative", "hard_negative"])
        self.assertEqual(
            result,
            ("hard_negative", 1, "ordinal_median_tiebreak", True),
        )

    def test_majority_vote_does_not_require_adjudication(self):
        result = consensus(["positive", "hard_negative", "positive"])
        self.assertEqual(result, ("positive", 2, "majority", False))

    def test_classification_metrics_include_confusion_and_kappa(self):
        metrics = classification_metrics(
            ["positive", "negative", "negative"],
            ["positive", "positive", "negative"],
            BINARY_LABELS,
        )
        self.assertEqual(metrics["matches"], 2)
        self.assertAlmostEqual(metrics["agreement"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"]["negative"]["positive"], 1)

    def test_perfect_single_class_agreement_has_unit_kappa(self):
        metrics = classification_metrics(
            ["negative", "negative"],
            ["negative", "negative"],
            BINARY_LABELS,
        )
        self.assertEqual(metrics["cohen_kappa"], 1.0)
        self.assertEqual(
            fleiss_kappa(
                [["hard_negative"] * 3, ["hard_negative"] * 3],
                THREE_LABELS,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
