import math
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
NUM_EPOCHS = 65
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

    splitter = GroupShuffleSplit(n_splits=500, test_size=0.2, random_state=42)
    best_split = None
    best_score = float("inf")

    for train_indices, validation_indices in splitter.split(
        np.zeros(len(labels)), labels, groups=session_ids
    ):
        train_counts = np.bincount(labels[train_indices], minlength=NUM_CLASSES)
        validation_counts = np.bincount(
            labels[validation_indices], minlength=NUM_CLASSES
        )
        if np.all(train_counts > 0) and np.all(validation_counts > 0):
            # Pick the session split closest to an 80/20 sample ratio
            # that maximizes the minimum training examples per letter
            val_ratio = len(validation_indices) / len(labels)
            ratio_penalty = abs(val_ratio - 0.22)
            min_train_penalty = 1.0 / max(int(train_counts.min()), 1)
            score = ratio_penalty + min_train_penalty
            if score < best_score:
                best_score = score
                best_split = (train_indices, validation_indices, class_counts)

    if best_split is not None:
        return best_split

    raise ValueError(
        "Could not make a session-separated split containing every letter in "
        "both training and validation. Record every letter in at least two "
        "different collection sessions (separate runs of collect_data.py)."
    )


def augment_batch(sequences):
    """Applies mild 3D wrist/camera tilt, timing shifts, scaling, and jitter."""
    batch_size, num_frames, num_features = sequences.shape
    device = sequences.device

    # 1. Temporal shift (-1 to +1 frames) to preserve fast Z/J trajectories
    shift = int(torch.randint(-1, 2, (1,)).item())
    if shift > 0 and num_frames > shift:
        pad = sequences[:, :1, :].expand(-1, shift, -1)
        sequences = torch.cat([pad, sequences[:, :-shift, :]], dim=1)
    elif shift < 0 and num_frames > -shift:
        s = -shift
        pad = sequences[:, -1:, :].expand(-1, s, -1)
        sequences = torch.cat([sequences[:, s:, :], pad], dim=1)

    # 2. 3D Hand & Camera Tilt Rotation across all 44 (x, y, z) vectors
    vectors = sequences.view(batch_size, num_frames, 44, 3)

    # Z-axis rotation: in-plane wrist tilt (+/- 12 degrees)
    theta_z = torch.empty(batch_size, device=device).uniform_(
        -math.radians(12), math.radians(12)
    )
    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)
    rot_z = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([sin_z,  cos_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)], dim=-1),
    ], dim=-2)

    # Y-axis rotation: horizontal camera angle turn (+/- 10 degrees)
    theta_y = torch.empty(batch_size, device=device).uniform_(
        -math.radians(10), math.radians(10)
    )
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
    rot_y = torch.stack([
        torch.stack([ cos_y, torch.zeros_like(cos_y), sin_y], dim=-1),
        torch.stack([torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y)], dim=-1),
        torch.stack([-sin_y, torch.zeros_like(cos_y), cos_y], dim=-1),
    ], dim=-2)

    rot_matrix = torch.bmm(rot_z, rot_y)
    vectors = torch.einsum("btvj,bjk->btvk", vectors, rot_matrix)
    sequences = vectors.reshape(batch_size, num_frames, num_features)

    # 3. Mild random scaling (+/- 8%) so O vs C gap stays crisp
    scale = torch.empty((batch_size, 1, 1), device=device).uniform_(0.92, 1.08)
    sequences = sequences * scale

    # 4. Subtle sensor jitter
    noise = torch.randn_like(sequences) * 0.005
    return sequences + noise


def train_architecture(
    architecture, train_loader, validation_loader, class_weights, device
):
    torch.manual_seed(42)
    model = create_sequence_model(architecture, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.03
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-4
    )

    best_accuracy = -1.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            sequences = augment_batch(sequences)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in validation_loader:
                sequences = sequences.to(
                    device, non_blocking=device.type == "cuda"
                )
                labels = labels.to(device, non_blocking=device.type == "cuda")
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
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] | "
                f"Train Loss: {average_loss:.4f} | "
                f"Val Acc: {validation_accuracy * 100:.2f}% | "
                f"LR: {current_lr:.5f}"
            )

    return best_accuracy, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training device: {torch.cuda.get_device_name(device)}")
    else:
        print("Training device: CPU")

    dataset_path = Path(__file__).resolve().with_name(SEQUENCE_CSV)
    dataset = SignSequenceDataset(dataset_path)
    train_indices, validation_indices, counts = split_by_session(
        dataset.labels.numpy(), dataset.session_ids
    )

    print(
        f"Loaded {len(dataset)} gesture sequences from {dataset_path.name} "
        f"({len(train_indices)} train, {len(validation_indices)} val) "
        f"across {len(np.unique(dataset.session_ids))} recording sessions."
    )
    print(
        "Examples per letter:",
        ", ".join(
            f"{chr(index + ord('A'))}:{count}"
            for index, count in enumerate(counts)
        ),
    )

    train_labels = dataset.labels.numpy()[train_indices]
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = train_counts.sum() / (NUM_CLASSES * np.maximum(train_counts, 1.0))
    class_weights = torch.tensor(np.power(weights, 0.75), dtype=torch.float32)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    results = []
    for architecture in ARCHITECTURES:
        print(f"\nTraining {architecture}...")
        accuracy, state = train_architecture(
            architecture, train_loader, validation_loader, class_weights, device
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