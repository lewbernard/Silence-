import tempfile
import unittest
from pathlib import Path

import numpy as np

from sequence_dataset import POSE_FEATURES_PER_FRAME
from thumb_gesture import (
    MIN_CAPTURES_PER_CLASS,
    OTHER_GESTURE_LABEL,
    THUMB_UP_LABEL,
    ThumbGestureClassifier,
    append_capture,
    train_thumb_classifier,
)


class ThumbGestureTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_path = self.root / "thumbs.csv"
        self.model_path = self.root / "thumbs.joblib"
        self.random = np.random.default_rng(42)

    def add_examples(self, label, center):
        for _ in range(MIN_CAPTURES_PER_CLASS):
            samples = self.random.normal(
                loc=center,
                scale=0.01,
                size=(5, POSE_FEATURES_PER_FRAME),
            ).astype(np.float32)
            append_capture(
                label,
                samples,
                f"session-{label}",
                self.dataset_path,
            )

    def test_trains_and_reloads_binary_thumb_classifier(self):
        self.add_examples(OTHER_GESTURE_LABEL, 0.0)
        self.add_examples(THUMB_UP_LABEL, 1.0)

        counts = train_thumb_classifier(self.dataset_path, self.model_path)
        classifier = ThumbGestureClassifier(self.model_path)

        self.assertEqual(counts[OTHER_GESTURE_LABEL], MIN_CAPTURES_PER_CLASS)
        self.assertEqual(counts[THUMB_UP_LABEL], MIN_CAPTURES_PER_CLASS)
        self.assertFalse(
            classifier.is_thumbs_up(
                np.zeros(POSE_FEATURES_PER_FRAME, dtype=np.float32)
            )
        )
        self.assertTrue(
            classifier.is_thumbs_up(
                np.ones(POSE_FEATURES_PER_FRAME, dtype=np.float32)
            )
        )

    def test_requires_multiple_captures_of_each_class(self):
        self.add_examples(THUMB_UP_LABEL, 1.0)

        with self.assertRaisesRegex(ValueError, "at least 3 separate examples"):
            train_thumb_classifier(self.dataset_path, self.model_path)

        self.assertFalse(self.model_path.exists())

    def test_rejects_invalid_sample_shape(self):
        with self.assertRaises(ValueError):
            append_capture(
                THUMB_UP_LABEL,
                np.zeros((3, 2), dtype=np.float32),
                "session",
                self.dataset_path,
            )


if __name__ == "__main__":
    unittest.main()
