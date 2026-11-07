from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from predict import LETTERS, MODEL_PATH
from sequence_dataset import NUM_CLASSES, SEQUENCE_CSV, SignSequenceDataset
from sequence_model import SignSequenceModel


def main():
    dataset_path = Path(__file__).resolve().with_name(SEQUENCE_CSV)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Sequence model not found: {MODEL_PATH}. Run train.py first."
        )

    dataset = SignSequenceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SignSequenceModel(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for sequences, batch_labels in loader:
            predictions.extend(model(sequences).argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())

    print("\n--- PER-LETTER SEQUENCE REPORT ---")
    print(
        classification_report(
            labels,
            predictions,
            labels=range(NUM_CLASSES),
            target_names=LETTERS,
            digits=3,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(labels, predictions, labels=range(NUM_CLASSES))
    confused_pairs = []
    for true_index in range(NUM_CLASSES):
        for predicted_index in range(NUM_CLASSES):
            if true_index != predicted_index and matrix[true_index, predicted_index]:
                confused_pairs.append(
                    (
                        matrix[true_index, predicted_index],
                        LETTERS[true_index],
                        LETTERS[predicted_index],
                    )
                )

    print("--- TOP CONFUSED LETTER PAIRS ---")
    if not confused_pairs:
        print("No confused letter pairs in this dataset.")
    else:
        for count, true_letter, predicted_letter in sorted(
            confused_pairs, reverse=True
        )[:10]:
            print(
                f"True '{true_letter}' mistaken for '{predicted_letter}' "
                f"-> {count} times"
            )


if __name__ == "__main__":
    main()
