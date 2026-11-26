import json
from pathlib import Path

import joblib
import numpy as np

from sequence_dataset import (
    FEATURES_PER_FRAME,
    NUM_FRAMES,
    POSE_FEATURES_PER_FRAME,
)


CUSTOM_SIGNS_PATH = Path(__file__).resolve().with_name("custom_signs.json")
CUSTOM_SIGN_MODEL_PATH = Path(__file__).resolve().with_name(
    "custom_sign_model.joblib"
)
FORMAT_VERSION = 1
MATCH_THRESHOLD = 0.25
SAMPLE_RATE = 15
CLASSIFIER_FORMAT_VERSION = 1
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.65
ACTION_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.28


def validate_sign_label(label):
    if not isinstance(label, str):
        raise ValueError("Custom sign labels must be text.")

    label = label.strip()
    if not label:
        raise ValueError("Custom sign labels must not be empty.")
    if len(label) > 40:
        raise ValueError("Custom sign labels must be 40 characters or fewer.")
    if any(ord(character) < 32 for character in label):
        raise ValueError("Custom sign labels must not contain control characters.")
    return label


class CustomSignLibrary:
    def __init__(self, path=CUSTOM_SIGNS_PATH):
        self.path = Path(path)
        self.signs = {}
        if self.path.is_file():
            self._load()

    def _load(self):
        with self.path.open(encoding="utf-8") as file:
            data = json.load(file)
        if (
            not isinstance(data, dict)
            or data.get("format_version") != FORMAT_VERSION
            or not isinstance(data.get("signs"), dict)
        ):
            raise ValueError(f"Invalid custom sign library format: {self.path}")

        for label, examples in data["signs"].items():
            label = validate_sign_label(label)
            if not isinstance(examples, list) or not examples:
                raise ValueError(
                    f"Custom sign {label!r} must contain at least one example."
                )
            self.signs[label] = [
                self._validate_sequence(example, label) for example in examples
            ]

    @staticmethod
    def _validate_sequence(sequence, label="sequence"):
        values = np.asarray(sequence, dtype=np.float32)
        expected_shape = (NUM_FRAMES, FEATURES_PER_FRAME)
        if values.shape != expected_shape:
            raise ValueError(
                f"Custom sign {label!r} must have shape {expected_shape}; "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Custom sign {label!r} contains non-finite values.")
        return values

    def add(self, label, sequence):
        label = validate_sign_label(label)
        values = self._validate_sequence(sequence, label)
        updated_signs = {
            name: list(examples) for name, examples in self.signs.items()
        }
        updated_signs.setdefault(label, []).append(values)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "signs": {
                name: [example.tolist() for example in examples]
                for name, examples in updated_signs.items()
            },
        }
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))
        self.signs = updated_signs
        if (
            self.path.resolve() == CUSTOM_SIGNS_PATH
            and CUSTOM_SIGN_MODEL_PATH.is_file()
        ):
            CUSTOM_SIGN_MODEL_PATH.unlink()

    def match(self, sequence):
        if not self.signs:
            return None

        values = self._validate_sequence(sequence)
        best_label = None
        best_distance = float("inf")
        for label, examples in self.signs.items():
            for example in examples:
                pose_distance = np.sqrt(
                    np.mean(
                        (
                            values[:, :POSE_FEATURES_PER_FRAME]
                            - example[:, :POSE_FEATURES_PER_FRAME]
                        )
                        ** 2,
                    )
                )
                velocity_distance = np.sqrt(
                    np.mean(
                        (
                            values[:, POSE_FEATURES_PER_FRAME:]
                            - example[:, POSE_FEATURES_PER_FRAME:]
                        )
                        ** 2,
                    )
                ) / SAMPLE_RATE
                distance = 0.7 * pose_distance + 0.3 * velocity_distance
                if distance < best_distance:
                    best_label = label
                    best_distance = distance

        if best_distance <= MATCH_THRESHOLD:
            return best_label
        return None


class CustomSignClassifier:
    def __init__(self, path=CUSTOM_SIGN_MODEL_PATH):
        self.path = Path(path)
        self.model = None
        self.labels = ()
        self.model_type = None
        self.acceptance_threshold = 0.0
        if self.path.is_file():
            self._load()

    def _load(self):
        checkpoint = joblib.load(self.path)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("format_version") != CLASSIFIER_FORMAT_VERSION
            or checkpoint.get("input_features") != NUM_FRAMES * FEATURES_PER_FRAME
            or not isinstance(checkpoint.get("labels"), list)
            or checkpoint.get("model_type") not in {"svm", "one_class_svm"}
        ):
            raise ValueError(f"Invalid custom sign classifier format: {self.path}")

        self.model = checkpoint["model"]
        threshold = checkpoint.get("acceptance_threshold", 0.0)
        if threshold is not None:
            threshold = float(threshold)
            if not np.isfinite(threshold):
                raise ValueError(
                    f"Invalid custom sign classifier threshold: {self.path}"
                )
            self.acceptance_threshold = threshold
        self.labels = tuple(
            validate_sign_label(label) for label in checkpoint["labels"]
        )
        self.model_type = checkpoint["model_type"]

    def predict(self, sequence):
        if self.model is None:
            return None

        values = CustomSignLibrary._validate_sequence(sequence)
        flattened = values.reshape(1, -1)
        if self.model_type == "one_class_svm":
            score = float(self.model.decision_function(flattened)[0])
            return (
                self.labels[0]
                if score >= self.acceptance_threshold
                else None
            )

        probabilities = self.model.predict_proba(flattened)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = str(self.model.classes_[predicted_index])
        confidence_threshold = CLASSIFIER_CONFIDENCE_THRESHOLD
        if (
            len(predicted_label) == 1
            and predicted_label in "0123456789"
        ) or predicted_label.casefold() in {"del", "delete"}:
            confidence_threshold = ACTION_CLASSIFIER_CONFIDENCE_THRESHOLD
        if probabilities[predicted_index] < confidence_threshold:
            number_candidates = [
                (index, probability)
                for index, (label, probability) in enumerate(
                    zip(self.model.classes_, probabilities)
                )
                if len(str(label)) == 1
                and str(label) in "0123456789"
                and probability >= ACTION_CLASSIFIER_CONFIDENCE_THRESHOLD
            ]
            if not number_candidates:
                return None
            predicted_index = max(
                number_candidates, key=lambda candidate: candidate[1]
            )[0]
            predicted_label = str(self.model.classes_[predicted_index])
        return predicted_label
