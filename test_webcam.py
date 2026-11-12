import unittest
from types import SimpleNamespace

from webcam import RecognizedSequence, is_thumbs_up


class RecognizedSequenceTests(unittest.TestCase):
    def test_confirmed_word_remains_after_spelling_prefix_is_cleared(self):
        sequence = RecognizedSequence()

        self.assertFalse(sequence.observe("A", 1.0))
        self.assertTrue(sequence.observe("A", 1.61))
        self.assertEqual(sequence.text, "a")
        self.assertFalse(sequence.observe("B", 1.62))
        self.assertTrue(sequence.observe("B", 2.23))
        self.assertEqual(sequence.current_word, "ab")
        self.assertEqual(sequence.text, "ab")

        sequence.confirm_word("about")

        self.assertEqual(sequence.text, "about")
        self.assertEqual(sequence.current_word, "")

    def test_cannot_confirm_word_that_does_not_match_prefix(self):
        sequence = RecognizedSequence()
        sequence.letters.append("A")

        with self.assertRaises(ValueError):
            sequence.confirm_word("hello")

    def test_keeps_unconfirmed_letters_in_buffer_after_decision_timeout(self):
        sequence = RecognizedSequence()
        sequence.letters.extend(("C", "A", "T"))

        sequence.keep_letters()

        self.assertEqual(sequence.text, "cat")
        self.assertEqual(sequence.current_word, "")


class ThumbsUpDetectionTests(unittest.TestCase):
    @staticmethod
    def make_landmarks():
        landmarks = [SimpleNamespace(y=0.5) for _ in range(21)]
        landmarks[2].y = 0.4
        landmarks[3].y = 0.3
        landmarks[4].y = 0.2
        for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20)):
            landmarks[pip].y = 0.5
            landmarks[tip].y = 0.7
        return landmarks

    def test_detects_raised_thumb_with_other_fingers_folded(self):
        self.assertTrue(is_thumbs_up(self.make_landmarks()))

    def test_rejects_pose_with_an_extended_finger(self):
        landmarks = self.make_landmarks()
        landmarks[8].y = 0.4

        self.assertFalse(is_thumbs_up(landmarks))


if __name__ == "__main__":
    unittest.main()
