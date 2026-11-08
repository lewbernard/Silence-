import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_signs import CustomSignLibrary
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


if __name__ == "__main__":
    unittest.main()
