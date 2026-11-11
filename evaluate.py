from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

from sequence_dataset import NUM_CLASSES, SEQUENCE_CSV, SignSequenceDataset
from sequence_model import create_sequence_model
from train import split_by_session, MODEL_PATH

LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)][:NUM_CLASSES]


def evaluate_model():
    dataset_path = Path(__file__).resolve().with_name(SEQUENCE_CSV)
    dataset = SignSequenceDataset(dataset_path)

    # Use the exact same unseen session split as train.py
    _, validation_indices, _ = split_by_session(
        dataset.labels.numpy(), dataset.session_ids
    )
    val_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=32, shuffle=False
    )

    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    architecture = checkpoint["architecture"]
    model = create_sequence_model(architecture, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print(f"\n--- DIAGNOSTIC REPORT ({architecture.upper()} ON UNSEEN SESSIONS) ---")
    print(classification_report(all_labels, all_preds, target_names=LETTERS, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    confused_pairs = []
    for true_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confused_pairs.append(
                    (cm[true_idx, pred_idx], LETTERS[true_idx], LETTERS[pred_idx])
                )

    confused_pairs.sort(reverse=True)
    print("--- TOP 10 CONFUSED LETTER PAIRS ---")
    for count, true_let, pred_let in confused_pairs[:10]:
        print(f"True '{true_let}' mistaken for '{pred_let}' -> {count} times")


if __name__ == "__main__":
    evaluate_model()