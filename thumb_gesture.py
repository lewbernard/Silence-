import csv
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sequence_dataset import POSE_FEATURES_PER_FRAME


THUMB_DATASET_PATH = Path(__file__).resolve().with_name(
    "thumb_gesture_dataset.csv"
)
THUMB_MODEL_PATH = Path(__file__).resolve().with_name(
    "thumb_gesture_model.joblib"
)
FORMAT_VERSION = 1
MIN_CAPTURES_PER_CLASS = 3
THUMB_UP_LABEL = 1
OTHER_GESTURE_LABEL = 0
THUMB_UP_CONFIDENCE_THRESHOLD = 0.75


def append_capture(label, samples, session_id, path=THUMB_DATASET_PATH):
    if label not in {OTHER_GESTURE_LABEL, THUMB_UP_LABEL}:
        raise ValueError(f"Invalid thumb gesture label: {label!r}.")

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != POSE_FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected samples with {POSE_FEATURES_PER_FRAME} features; "
            f"got shape {values.shape}."
        )
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("A capture must contain finite hand landmark samples.")
    if not session_id:
        raise ValueError("Session ID must not be empty.")

    capture_id = uuid.uuid4().hex
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for sample in values:
            writer.writerow(
                [label, session_id, capture_id, *sample.tolist()]
            )
    return capture_id


def load_captures(path=THUMB_DATASET_PATH):
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Thumb training data not found: {dataset_path}.")

    features = []
    labels = []
    capture_ids = []
    expected_columns = POSE_FEATURES_PER_FRAME + 3
    with dataset_path.open(newline="", encoding="utf-8") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if len(row) != expected_columns:
                raise ValueError(
                    f"{dataset_path}, row {line_number}: expected "
                    f"{expected_columns} columns; found {len(row)}."
                )
            try:
                label = int(row[0])
                values = np.asarray(row[3:], dtype=np.float32)
            except ValueError as error:
                raise ValueError(
                    f"{dataset_path}, row {line_number}: label and features "
                    "must be numeric."
                ) from error

            if label not in {OTHER_GESTURE_LABEL, THUMB_UP_LABEL}:
                raise ValueError(
                    f"{dataset_path}, row {line_number}: label must be 0 or 1."
                )
            if not row[1] or not row[2]:
                raise ValueError(
                    f"{dataset_path}, row {line_number}: session and capture "
                    "IDs must not be empty."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{dataset_path}, row {line_number}: features contain "
                    "non-finite values."
                )
            features.append(values)
            labels.append(label)
            capture_ids.append(row[2])

    if not labels:
        raise ValueError(f"No thumb training samples found in {dataset_path}.")
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(capture_ids),
    )


def train_thumb_classifier(
    dataset_path=THUMB_DATASET_PATH, model_path=THUMB_MODEL_PATH
):
    features, labels, capture_ids = load_captures(dataset_path)
    captures_per_class = {
        label: len(set(capture_ids[labels == label]))
        for label in (OTHER_GESTURE_LABEL, THUMB_UP_LABEL)
    }
    insufficient = {
        label: count
        for label, count in captures_per_class.items()
        if count < MIN_CAPTURES_PER_CLASS
    }
    if insufficient:
        names = {
            OTHER_GESTURE_LABEL: "other hand pose",
            THUMB_UP_LABEL: "thumbs-up",
        }
        details = ", ".join(
            f"{names[label]}: {count}"
            for label, count in insufficient.items()
        )
        raise ValueError(
            f"Capture at least {MIN_CAPTURES_PER_CLASS} separate examples of "
            f"each class before training. Need more for {details}."
        )

    model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", class_weight="balanced"),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    )
    model.fit(features, labels)
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "input_features": POSE_FEATURES_PER_FRAME,
        "model": model,
    }
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, model_file)
    return captures_per_class


class ThumbGestureClassifier:
    def __init__(self, path=THUMB_MODEL_PATH):
        self.path = Path(path)
        self.model = None
        if self.path.is_file():
            checkpoint = joblib.load(self.path)
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("format_version") != FORMAT_VERSION
                or checkpoint.get("input_features") != POSE_FEATURES_PER_FRAME
                or "model" not in checkpoint
            ):
                raise ValueError(f"Invalid thumb gesture model: {self.path}")
            self.model = checkpoint["model"]
            if not np.array_equal(
                self.model.classes_,
                np.asarray((OTHER_GESTURE_LABEL, THUMB_UP_LABEL)),
            ):
                raise ValueError(
                    f"Thumb gesture model must classify both hand-pose labels: "
                    f"{self.path}"
                )

    def is_thumbs_up(self, features):
        if self.model is None:
            return None

        values = np.asarray(features, dtype=np.float32)
        if values.shape != (POSE_FEATURES_PER_FRAME,):
            raise ValueError(
                f"Expected {POSE_FEATURES_PER_FRAME} hand features; "
                f"got shape {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Hand features contain non-finite values.")

        probabilities = self.model.predict_proba(values.reshape(1, -1))[0]
        positive_index = int(
            np.flatnonzero(self.model.classes_ == THUMB_UP_LABEL)[0]
        )
        return (
            float(probabilities[positive_index])
            >= THUMB_UP_CONFIDENCE_THRESHOLD
        )
