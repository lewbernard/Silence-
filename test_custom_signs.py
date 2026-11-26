import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_signs import CustomSignClassifier, CustomSignLibrary
from sequence_dataset import FEATURES_PER_FRAME, NUM_FRAMES, POSE_FEATURES_PER_FRAME


class CustomSignLibraryTests(unittest.TestCase):
    def test_saves_and_reloads_custom_sign_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_signs.json"
            library = CustomSignLibrary(path)
            sequence = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

            library.add("hello", sequence)
            library.add("hello", sequence)

            loaded_library = CustomSignLibrary(path)
            self.assertEqual(len(loaded_library.signs["hello"]), 2)
            self.assertEqual(loaded_library.match(sequence), "hello")

    def test_matches_nearby_sequence_and_rejects_different_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            library = CustomSignLibrary(Path(directory) / "custom_signs.json")
            sequence = np.random.default_rng(42).normal(
                size=(NUM_FRAMES, FEATURES_PER_FRAME)
            ).astype(np.float32)
            library.add("hello", sequence)

            nearby_sequence = sequence + np.random.default_rng(43).normal(
                scale=0.01, size=sequence.shape
            ).astype(np.float32)
            different_sequence = sequence.copy()
            different_sequence[:, :POSE_FEATURES_PER_FRAME] += 1

            self.assertEqual(library.match(nearby_sequence), "hello")
            self.assertIsNone(library.match(different_sequence))

    def test_rejects_invalid_sequence_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            library = CustomSignLibrary(Path(directory) / "custom_signs.json")

            with self.assertRaises(ValueError):
                library.add("hello", np.zeros((2, 2), dtype=np.float32))


class CustomSignClassifierConfidenceTests(unittest.TestCase):
    class ProbabilityModel:
        def __init__(self, labels, probabilities):
            self.classes_ = np.asarray(labels)
            self.probabilities = probabilities

        def predict_proba(self, values):
            return np.tile(self.probabilities, (len(values), 1))

    @staticmethod
    def classifier_for(labels, probabilities):
        classifier = CustomSignClassifier(path="unused.joblib")
        classifier.model = CustomSignClassifierConfidenceTests.ProbabilityModel(
            labels, probabilities
        )
        classifier.model_type = "svm"
        return classifier

    def test_accepts_low_confidence_number_prediction_for_hold_resolution(self):
        classifier = self.classifier_for(
            ["0", "other-1", "other-2", "other-3", "other-4"],
            [0.3, 0.175, 0.175, 0.175, 0.175],
        )
        sequence = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

        self.assertEqual(classifier.predict(sequence), "0")

    def test_prefers_number_candidate_over_confused_letter_class(self):
        classifier = self.classifier_for(
            ["o", "0", "other-1", "other-2", "other-3", "other-4"],
            [0.53, 0.3, 0.05, 0.05, 0.04, 0.03],
        )
        sequence = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

        self.assertEqual(classifier.predict(sequence), "0")

    def test_accepts_low_confidence_delete_alias(self):
        classifier = self.classifier_for(
            ["del", "other-1", "other-2", "other-3", "other-4"],
            [0.3, 0.175, 0.175, 0.175, 0.175],
        )
        sequence = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

        self.assertEqual(classifier.predict(sequence), "del")

    def test_keeps_standard_threshold_for_other_custom_signs(self):
        classifier = self.classifier_for(
            ["hello", "other-1", "other-2", "other-3", "other-4"],
            [0.3, 0.175, 0.175, 0.175, 0.175],
        )
        sequence = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

        self.assertIsNone(classifier.predict(sequence))


if __name__ == "__main__":
    unittest.main()
