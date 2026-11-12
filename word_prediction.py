import json
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path


DICTIONARY_PATH = Path(__file__).resolve().with_name("english_words.txt")
FEEDBACK_PATH = Path(__file__).resolve().with_name("word_feedback.jsonl")
COMMON_WORDS = (
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any",
    "these", "give", "day", "most", "us", "hello", "help", "here",
)
WORD_DECISION_SECONDS = 5.0
SUGGESTION_COUNT = 3


class WordPredictor:
    def __init__(self, dictionary_path=DICTIONARY_PATH, feedback_path=FEEDBACK_PATH):
        self.dictionary_path = Path(dictionary_path)
        self.feedback_path = Path(feedback_path)
        self.words = self._load_dictionary()
        self.word_counts = Counter()
        self._load_feedback()
        self.common_word_ranks = {
            word: index for index, word in enumerate(COMMON_WORDS)
        }

    def _load_dictionary(self):
        if not self.dictionary_path.is_file():
            raise FileNotFoundError(
                f"English dictionary not found: {self.dictionary_path}. "
                "Restore english_words.txt from the project or download the "
                "public-domain word list documented in README.md."
            )

        words = set()
        with self.dictionary_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                word = line.strip().lower()
                if word and word.isascii() and word.isalpha():
                    words.add(word)
                elif word:
                    raise ValueError(
                        f"{self.dictionary_path}, line {line_number}: "
                        "dictionary entries must contain only English letters."
                    )

        if not words:
            raise ValueError(f"No English words found in {self.dictionary_path}.")
        return tuple(sorted(words))

    def _load_feedback(self):
        if not self.feedback_path.is_file():
            return

        with self.feedback_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{self.feedback_path}, line {line_number}: "
                        "invalid feedback JSON."
                    ) from error

                if not isinstance(event, dict):
                    raise ValueError(
                        f"{self.feedback_path}, line {line_number}: "
                        "feedback entries must be JSON objects."
                    )
                prefix = event.get("prefix")
                accepted_word = event.get("accepted_word")
                if (
                    not isinstance(prefix, str)
                    or not prefix.isascii()
                    or not prefix.isalpha()
                    or not isinstance(accepted_word, str)
                    or not accepted_word.isascii()
                    or not accepted_word.isalpha()
                    or not accepted_word.startswith(prefix)
                ):
                    raise ValueError(
                        f"{self.feedback_path}, line {line_number}: "
                        "feedback must contain a matching prefix and accepted word."
                    )
                self.word_counts[accepted_word] += 1

    def suggest(self, prefix, limit=SUGGESTION_COUNT):
        prefix = prefix.strip().lower()
        if not prefix or not prefix.isascii() or not prefix.isalpha():
            return []
        if limit < 1:
            raise ValueError("Suggestion limit must be positive.")

        start = bisect_left(self.words, prefix)
        matches = []
        for word in islice(self.words, start, None):
            if not word.startswith(prefix):
                break
            if len(word) >= len(prefix):
                matches.append(word)
        matches.sort(
            key=lambda word: (
                -self.word_counts[word],
                self.common_word_ranks.get(word, len(COMMON_WORDS)),
                len(word),
                word,
            )
        )
        return matches[:limit]

    def record_confirmation(self, prefix, suggestions, accepted_word):
        prefix = prefix.strip().lower()
        accepted_word = accepted_word.strip().lower()
        if not prefix.isascii() or not prefix.isalpha():
            raise ValueError("Confirmed word prefix must contain English letters.")
        if not accepted_word.isascii() or not accepted_word.isalpha():
            raise ValueError("Confirmed words must contain English letters.")
        if not accepted_word.startswith(prefix):
            raise ValueError(
                f"Confirmed word {accepted_word!r} does not match prefix "
                f"{prefix!r}."
            )
        if accepted_word not in suggestions:
            raise ValueError("Confirmed word must be one of the displayed suggestions.")

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prefix": prefix,
            "suggestions": list(suggestions),
            "accepted_word": accepted_word,
            "confirmation": "thumbs_up",
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=True) + "\n")
        self.word_counts[accepted_word] += 1
        return event
