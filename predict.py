from pathlib import Path

import numpy as np
import torch

from sequence_dataset import FEATURES_PER_FRAME, NUM_CLASSES, NUM_FRAMES
from sequence_model import SignSequenceModel


LETTERS = [chr(index) for index in range(ord("A"), ord("Z") + 1)]
MODEL_PATH = Path(__file__).resolve().with_name("asl_sequence_model_fast.pth")


class SignPredictor:
    def __init__(self, model_filename=MODEL_PATH):
        model_path = Path(model_filename)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Sequence model not found: {model_path}. "
                "Record fast gesture sequences and run train.py first."
            )

        self.model = SignSequenceModel(num_classes=NUM_CLASSES)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, sequence):
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
            confidence, predicted_index = probabilities.max(dim=1)

        return LETTERS[predicted_index.item()], confidence.item() * 100
