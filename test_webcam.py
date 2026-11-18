import unittest
from types import SimpleNamespace

from webcam import (
    NumberSignResolver,
    RecognizedSequence,
    infer_number_letter_conflicts,
    is_thumbs_up,
    matching_number_candidate,
)


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

    def test_delete_removes_latest_letter_then_latest_word(self):
        sequence = RecognizedSequence()
        sequence.letters.extend(("C", "A"))

        sequence.commit("delete")
        self.assertEqual(sequence.current_word, "c")

        previous_word = RecognizedSequence()
        previous_word.confirmed_words.append("hello")
        previous_word.commit("delete")
        self.assertEqual(previous_word.text, "")

    def test_number_is_a_separate_text_item_and_can_be_deleted(self):
        sequence = RecognizedSequence()
        sequence.letters.extend(("C", "A", "T"))

        sequence.commit("2")

        self.assertEqual(sequence.text, "cat 2")
        self.assertEqual(sequence.current_word, "")
        sequence.commit("delete")
        self.assertEqual(sequence.text, "cat")


class NumberSignResolverTests(unittest.TestCase):
    def test_shows_asl_letter_until_conflicting_number_hold_expires(self):
        resolver = NumberSignResolver()

        resolved, shown, pending = resolver.update("1", "D", 1.0)
        self.assertEqual((resolved, shown, pending), ([], "D", True))

        resolved, shown, pending = resolver.update("1", "D", 2.49)
        self.assertEqual((resolved, shown, pending), ([], "D", True))

        resolved, shown, pending = resolver.update("1", "D", 2.5)
        self.assertEqual((resolved, shown, pending), (["1"], "1", True))

    def test_sign_change_resolves_pending_number_as_letter(self):
        resolver = NumberSignResolver()
        resolver.update("0", "O", 1.0)

        resolved, shown, pending = resolver.update(None, "B", 1.5)

        self.assertEqual((resolved, shown, pending), (["O"], "B", False))


class NumberLetterConflictTests(unittest.TestCase):
    def test_infers_consistent_letter_for_a_trained_number(self):
        class Predictor:
            def predict(self, sample):
                return sample, 100.0

        conflicts = infer_number_letter_conflicts(
            {"0": ["O", "O", "O", "A", "A"], "love": ["A"]},
            Predictor(),
        )

        self.assertEqual(conflicts, {"0": "O"})

    def test_ignores_number_prediction_for_a_different_letter(self):
        conflicts = {"0": "O", "1": "D"}

        self.assertEqual(matching_number_candidate("0", "O", conflicts), "0")
        self.assertIsNone(matching_number_candidate("0", "A", conflicts))


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
