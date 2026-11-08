import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_signs import CustomSignClassifier
from custom_train import (
    MIN_EXAMPLES_PER_SIGN,
    train_and_save_custom_classifier,
    train_custom_classifier,
)
from sequence_dataset import FEATURES_PER_FRAME, NUM_FRAMES


class CustomSignTrainingTests(unittest.TestCase):
    def _examples(self, offset, count=MIN_EXAMPLES_PER_SIGN):
        rng = np.random.default_rng(42 + offset)
        return [
            (
                np.full((NUM_FRAMES, FEATURES_PER_FRAME), offset, dtype=np.float32)
                + rng.normal(
                    scale=0.01,
                    size=(NUM_FRAMES, FEATURES_PER_FRAME),
                ).astype(np.float32)
            )
            for _ in range(count)
        ]

    def test_trains_classifier_for_one_custom_sign(self):
        signs = {"hello": self._examples(0, count=10)}

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom_sign_model.joblib"
            _, counts = train_and_save_custom_classifier(
                signs, model_path
            )
            classifier = CustomSignClassifier(model_path)

            self.assertEqual(counts["hello"], 10)
            self.assertEqual(classifier.predict(signs["hello"][0]), "hello")
            self.assertIsNone(
                classifier.predict(
                    np.full(
                        (NUM_FRAMES, FEATURES_PER_FRAME),
                        10,
                        dtype=np.float32,
                    )
                )
            )

    def test_trains_classifier_for_multiple_custom_signs(self):
        signs = {"hello": self._examples(0), "thanks": self._examples(2)}
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom_sign_model.joblib"
            train_and_save_custom_classifier(signs, model_path)
            classifier = CustomSignClassifier(model_path)

            self.assertEqual(classifier.predict(signs["hello"][0]), "hello")
            self.assertEqual(classifier.predict(signs["thanks"][0]), "thanks")

    def test_requires_minimum_examples_per_sign(self):
        with self.assertRaisesRegex(ValueError, "at least 5 examples per sign"):
            train_custom_classifier(
                {"hello": [np.zeros((NUM_FRAMES, FEATURES_PER_FRAME))]}
            )


if __name__ == "__main__":
    unittest.main()
