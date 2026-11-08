from collections import Counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

from custom_signs import (
    CLASSIFIER_FORMAT_VERSION,
    CUSTOM_SIGN_MODEL_PATH,
    CustomSignLibrary,
)
from sequence_dataset import FEATURES_PER_FRAME, NUM_FRAMES


MIN_EXAMPLES_PER_SIGN = 5


def train_custom_classifier(signs):
    counts = Counter(
        label for label, examples in signs.items() for _ in examples
    )
    if not counts:
        raise ValueError(
            "No custom signs found. Run webcam.py and record examples first."
        )

    insufficient = {
        label: count
        for label, count in counts.items()
        if count < MIN_EXAMPLES_PER_SIGN
    }
    if insufficient:
        details = ", ".join(
            f"{label!r}: {count}" for label, count in sorted(insufficient.items())
        )
        raise ValueError(
            f"Record at least {MIN_EXAMPLES_PER_SIGN} examples per sign before "
            f"training. Need more for {details}."
        )

    labels = []
    sequences = []
    for label, examples in signs.items():
        for example in examples:
            labels.append(label)
            sequences.append(
                CustomSignLibrary._validate_sequence(example, label).reshape(-1)
            )

    features = np.stack(sequences)
    acceptance_threshold = None
    if len(counts) == 1:
        model_type = "one_class_svm"
        model = make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
        )
        model.fit(features)
        training_scores = model.decision_function(features)
        acceptance_threshold = float(np.quantile(training_scores, 0.05))
    else:
        model_type = "svm"
        model = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", class_weight="balanced"),
                method="sigmoid",
                cv=5,
                ensemble=False,
            ),
        )
        model.fit(features, labels)

    return {
        "format_version": CLASSIFIER_FORMAT_VERSION,
        "model_type": model_type,
        "labels": sorted(counts),
        "input_features": NUM_FRAMES * FEATURES_PER_FRAME,
        "acceptance_threshold": acceptance_threshold,
        "model": model,
    }, counts


def train_and_save_custom_classifier(signs, path=CUSTOM_SIGN_MODEL_PATH):
    checkpoint, counts = train_custom_classifier(signs)
    joblib.dump(checkpoint, path)
    return checkpoint, counts


def main():
    library = CustomSignLibrary()
    checkpoint, counts = train_and_save_custom_classifier(library.signs)

    print(
        f"Trained custom-only {checkpoint['model_type']} classifier with "
        f"{sum(counts.values())} examples across {len(counts)} signs."
    )
    print(
        "Examples per sign: "
        + ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    )
    print(f"Saved custom classifier to {CUSTOM_SIGN_MODEL_PATH.name}.")
    print("Restart webcam.py to use the trained custom-sign classifier.")


if __name__ == "__main__":
    main()
