import unittest
from types import SimpleNamespace

from webcam import (
    NumberSignResolver,
    RecognizedSequence,
    is_thumbs_up,
    normalize_custom_sign_label,
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

    def test_del_alias_is_canonicalized_as_delete(self):
        sequence = RecognizedSequence()
        sequence.letters.extend(("C", "A"))

        sequence.commit(normalize_custom_sign_label("del"))

        self.assertEqual(sequence.current_word, "c")

    def test_holding_delete_repeats_once_per_two_seconds(self):
        sequence = RecognizedSequence()
        sequence.confirmed_words.append("hello")
        sequence.letters.extend(("C", "A"))

        self.assertFalse(sequence.observe("delete", 1.0))
        self.assertFalse(sequence.observe("delete", 2.99))
        self.assertTrue(sequence.observe("delete", 3.0))
        self.assertEqual(sequence.current_word, "c")

        self.assertFalse(sequence.observe("del", 4.99))
        self.assertTrue(sequence.observe("delete", 5.0))
        self.assertEqual(sequence.text, "hello")

        self.assertFalse(sequence.observe("delete", 6.99))
        self.assertTrue(sequence.observe("delete", 7.0))
        self.assertEqual(sequence.text, "")

    def test_keep_letters_does_not_restart_an_active_delete_hold(self):
        sequence = RecognizedSequence()
        sequence.letters.append("A")
        sequence.observe("delete", 1.0)

        sequence.keep_letters()

        self.assertEqual(sequence.delete_hold_remaining(2.0), 1.0)
        self.assertTrue(sequence.observe("delete", 3.0))
        self.assertEqual(sequence.text, "")

    def test_delete_countdown_is_safe_without_an_active_timer(self):
        sequence = RecognizedSequence()

        self.assertIsNone(sequence.delete_hold_remaining(1.0))

    def test_brief_hand_tracking_gap_does_not_restart_delete_hold(self):
        sequence = RecognizedSequence()
        sequence.letters.append("A")

        self.assertFalse(sequence.observe("delete", 1.0))
        self.assertFalse(sequence.observe(None, 2.9))
        self.assertTrue(sequence.observe("delete", 3.0))
        self.assertEqual(sequence.current_word, "")

    def test_delete_aliases_share_a_stable_prediction_label(self):
        self.assertEqual(normalize_custom_sign_label("del"), "delete")
        self.assertEqual(normalize_custom_sign_label("DELETE"), "delete")

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

        resolved, shown, pending = resolver.update("1", "D", 2.99)
        self.assertEqual((resolved, shown, pending), ([], "D", True))

        resolved, shown, pending = resolver.update("1", "D", 3.0)
        self.assertEqual((resolved, shown, pending), (["1"], "1", True))

    def test_sign_change_resolves_pending_number_as_letter(self):
        resolver = NumberSignResolver()
        resolver.update("0", "O", 1.0)

        resolved, shown, pending = resolver.update(None, "B", 1.5)

        self.assertEqual((resolved, shown, pending), (["O"], "B", False))


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
