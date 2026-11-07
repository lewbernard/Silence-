from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from sequence_dataset import NUM_CLASSES, SEQUENCE_CSV, SignSequenceDataset
from sequence_model import create_sequence_model


MODEL_PATH = Path(__file__).resolve().with_name("asl_motion_model.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 30
ARCHITECTURES = ("gru", "temporal_cnn")


def split_by_session(labels, session_ids):
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    missing = [
        chr(index + ord("A"))
        for index, count in enumerate(class_counts)
        if count == 0
    ]
    if missing:
        raise ValueError(
            "The sequence dataset is missing examples for: "
            + ", ".join(missing)
            + ". Record examples for all 26 letters."
        )

    session_count = len(np.unique(session_ids))
    if session_count < 2:
        raise ValueError(
            "Session-based validation needs at least two collection sessions. "
            "Run collect_data.py again and record all 26 letters in each run."
        )

    splitter = GroupShuffleSplit(n_splits=200, test_size=0.2, random_state=42)
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(labels)), labels, groups=session_ids
    ):
        train_counts = np.bincount(labels[train_indices], minlength=NUM_CLASSES)
        validation_counts = np.bincount(
            labels[validation_indices], minlength=NUM_CLASSES
        )
        if np.all(train_counts > 0) and np.all(validation_counts > 0):
            return train_indices, validation_indices, class_counts

    raise ValueError(
        "Could not make a session-separated split containing every letter in "
        "both training and validation. Record every letter in at least two "
        "different collection sessions (separate runs of collect_data.py)."
    )


def train_architecture(architecture, train_loader, validation_loader):
    torch.manual_seed(42)
    model = create_sequence_model(architecture, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = -1.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in validation_loader:
                predictions = model(sequences).argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        validation_accuracy = correct / total
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            average_loss = running_loss / len(train_loader)
            print(
                f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] | "
                f"Training Loss: {average_loss:.4f} | "
                f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
            )

    return best_accuracy, best_state


def main():
    dataset_path = Path(__file__).resolve().with_name(SEQUENCE_CSV)
    dataset = SignSequenceDataset(dataset_path)
    train_indices, validation_indices, counts = split_by_session(
        dataset.labels.numpy(), dataset.session_ids
    )

    print(
        f"Loaded {len(dataset)} gesture sequences from {dataset_path.name} "
        f"across {len(np.unique(dataset.session_ids))} recording sessions."
    )
    print(
        "Examples per letter:",
        ", ".join(
            f"{chr(index + ord('A'))}:{count}"
            for index, count in enumerate(counts)
        ),
    )
    if counts.min() < 20:
        print(
            "Warning: some letters have fewer than 20 examples; "
            "collecting more varied examples should improve reliability."
        )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []
    for architecture in ARCHITECTURES:
        print(f"\nTraining {architecture}...")
        accuracy, state = train_architecture(
            architecture, train_loader, validation_loader
        )
        results.append((accuracy, architecture, state))
        print(
            f"Best session-validation accuracy for {architecture}: "
            f"{accuracy * 100:.2f}%"
        )

    best_accuracy, best_architecture, best_state = max(
        results, key=lambda item: item[0]
    )
    torch.save(
        {
            "architecture": best_architecture,
            "state_dict": best_state,
            "validation_accuracy": best_accuracy,
            "num_frames": dataset.features.shape[1],
            "input_features": dataset.features.shape[2],
        },
        MODEL_PATH,
    )
    print(
        f"\nSelected {best_architecture} using session-separated validation "
        f"({best_accuracy * 100:.2f}%). Saved to {MODEL_PATH.name}."
    )


if __name__ == "__main__":
    main()
