from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sequence_dataset import NUM_CLASSES, SEQUENCE_CSV, SignSequenceDataset
from sequence_model import SignSequenceModel


MODEL_PATH = Path(__file__).resolve().with_name("asl_sequence_model_fast.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 30


def split_by_class(labels):
    rng = np.random.default_rng(42)
    train_indices = []
    validation_indices = []
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    missing = [chr(index + ord("A")) for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(
            "The sequence dataset is missing examples for: "
            + ", ".join(missing)
            + ". Record sequences for all 26 letters before training."
        )

    for label, count in enumerate(counts):
        if count < 2:
            letter = chr(label + ord("A"))
            raise ValueError(
                f"Need at least 2 recorded sequences for {letter}; found {count}."
            )

        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_count = max(1, round(count * 0.2))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    return train_indices, validation_indices, counts


def main():
    dataset_path = Path(__file__).resolve().with_name(SEQUENCE_CSV)
    dataset = SignSequenceDataset(dataset_path)
    train_indices, validation_indices, counts = split_by_class(
        dataset.labels.numpy()
    )

    print(f"Loaded {len(dataset)} gesture sequences from {dataset_path.name}.")
    print("Examples per letter:", ", ".join(
        f"{chr(index + ord('A'))}:{count}" for index, count in enumerate(counts)
    ))
    if counts.min() < 20:
        print(
            "Warning: some letters have fewer than 20 examples; "
            "collecting more balanced examples may improve recognition."
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

    model = SignSequenceModel(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting 26-letter sequence training on CPU...")
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
                outputs = model(sequences)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        validation_accuracy = 100 * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            average_loss = running_loss / len(train_loader)
            print(
                f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] | "
                f"Training Loss: {average_loss:.4f} | "
                f"Validation Accuracy: {validation_accuracy:.2f}%"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete! Saved sequence model to {MODEL_PATH.name}.")


if __name__ == "__main__":
    main()
