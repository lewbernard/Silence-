import json
import tempfile
import unittest
from pathlib import Path

from word_prediction import WordPredictor


class WordPredictorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dictionary_path = self.root / "words.txt"
        self.feedback_path = self.root / "feedback.jsonl"
        self.dictionary_path.write_text(
            "there\ntheir\nthe\nthen\nhello\nhelp\nhelicopter\n",
            encoding="utf-8",
        )

    def test_suggests_dictionary_words_for_prefix(self):
        predictor = WordPredictor(self.dictionary_path, self.feedback_path)

        self.assertEqual(predictor.suggest("th"), ["the", "there", "their"])
        self.assertEqual(predictor.suggest("HEL"), ["hello", "help", "helicopter"])
        self.assertEqual(predictor.suggest("!"), [])

    def test_can_suggest_a_word_that_exactly_matches_the_spelled_prefix(self):
        predictor = WordPredictor(self.dictionary_path, self.feedback_path)

        self.assertEqual(predictor.suggest("hello"), ["hello"])

    def test_confirmation_is_saved_and_ranks_future_suggestions(self):
        predictor = WordPredictor(self.dictionary_path, self.feedback_path)
        suggestions = predictor.suggest("th")

        event = predictor.record_confirmation("th", suggestions, "their")
        reloaded = WordPredictor(self.dictionary_path, self.feedback_path)

        self.assertEqual(event["accepted_word"], "their")
        self.assertEqual(reloaded.suggest("th")[0], "their")
        with self.feedback_path.open(encoding="utf-8") as file:
            self.assertEqual(json.loads(file.readline()), event)

    def test_rejects_confirmation_outside_suggestions(self):
        predictor = WordPredictor(self.dictionary_path, self.feedback_path)

        with self.assertRaises(ValueError):
            predictor.record_confirmation("th", ["the"], "their")

    def test_rejects_malformed_feedback_instead_of_ignoring_it(self):
        self.feedback_path.write_text("{invalid json}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid feedback JSON"):
            WordPredictor(self.dictionary_path, self.feedback_path)


if __name__ == "__main__":
    unittest.main()
