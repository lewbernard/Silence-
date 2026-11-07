from pathlib import Path

import numpy as np
import torch

from sequence_dataset import (
    FEATURES_PER_FRAME,
    NUM_CLASSES,
    NUM_FRAMES,
)
from sequence_model import create_sequence_model


LETTERS = [chr(index) for index in range(ord("A"), ord("Z") + 1)]
MODEL_PATH = Path(__file__).resolve().with_name("asl_motion_model.pth")


class SignPredictor:
    def __init__(self, model_filename=MODEL_PATH):
        model_path = Path(model_filename)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Sequence model not found: {model_path}. "
                "Record motion sequences and run train.py first."
            )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("num_frames") != NUM_FRAMES:
            raise ValueError(
                f"Model expects {checkpoint.get('num_frames')} frames, "
                f"but this pipeline uses {NUM_FRAMES}."
            )
        if checkpoint.get("input_features") != FEATURES_PER_FRAME:
            raise ValueError(
                f"Model expects {checkpoint.get('input_features')} features "
                f"per frame, but this pipeline uses {FEATURES_PER_FRAME}."
            )
        self.model = create_sequence_model(
            checkpoint["architecture"], num_classes=NUM_CLASSES
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict_probabilities(self, sequence):
        values = np.asarray(sequence, dtype=np.float32)
        expected_shape = (NUM_FRAMES, FEATURES_PER_FRAME)
        if values.shape != expected_shape:
            raise ValueError(
                f"Expected a sequence with shape {expected_shape}; "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Gesture sequence contains non-finite values.")

        input_tensor = torch.from_numpy(values).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)
        return probabilities[0].numpy()

    def predict(self, sequence):
        probabilities = self.predict_probabilities(sequence)
        predicted_index = int(np.argmax(probabilities))
        return LETTERS[predicted_index], float(probabilities[predicted_index] * 100)
