import csv

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_CLASSES = 26
NUM_FRAMES = 15
POSE_FEATURES_PER_FRAME = 66
FEATURES_PER_FRAME = POSE_FEATURES_PER_FRAME * 2
SEQUENCE_FEATURES = NUM_FRAMES * FEATURES_PER_FRAME
SEQUENCE_CSV = "asl_motion_sequence_dataset.csv"


def normalize_hand_coordinates(coords_63):
    joints = coords_63.reshape(21, 3)
    centered_joints = joints - joints[0]
    max_value = np.max(np.abs(centered_joints))
    if max_value > 0:
        centered_joints = centered_joints / max_value
    return centered_joints.flatten().astype(np.float32)


def landmarks_to_features(landmarks):
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks; got {len(landmarks)}.")

    coordinates = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
        dtype=np.float32,
    )
    relative_hand = normalize_hand_coordinates(coordinates.reshape(-1))
    wrist_position = coordinates[0]
    return np.concatenate((relative_hand, wrist_position)).astype(np.float32)


def add_velocity_features(pose_sequence, sample_rate=15):
    poses = np.asarray(pose_sequence, dtype=np.float32)
    expected_shape = (NUM_FRAMES, POSE_FEATURES_PER_FRAME)
    if poses.shape != expected_shape:
        raise ValueError(f"Expected pose sequence shape {expected_shape}; got {poses.shape}.")
    velocities = np.zeros_like(poses)
    velocities[1:] = np.diff(poses, axis=0) * sample_rate
    return np.concatenate((poses, velocities), axis=1)


def load_sequence_csv(csv_file=SEQUENCE_CSV):
    sequences = []
    labels = []
    session_ids = []

    with open(csv_file, newline="", encoding="utf-8") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if len(row) != SEQUENCE_FEATURES + 2:
                raise ValueError(
                    f"{csv_file}, row {line_number}: expected "
                    f"{SEQUENCE_FEATURES + 2} columns (label, session ID, and "
                    f"{NUM_FRAMES} frames of {FEATURES_PER_FRAME} features); "
                    f"found {len(row)}."
                )

            try:
                label = int(row[0])
                values = np.asarray(row[2:], dtype=np.float32)
            except ValueError as error:
                raise ValueError(
                    f"{csv_file}, row {line_number}: label and features "
                    "must be numeric."
                ) from error

            if not 0 <= label < NUM_CLASSES:
                raise ValueError(
                    f"{csv_file}, row {line_number}: label {label} is outside "
                    f"the valid range 0-{NUM_CLASSES - 1}."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{csv_file}, row {line_number}: features contain "
                    "non-finite values."
                )
            if not row[1]:
                raise ValueError(
                    f"{csv_file}, row {line_number}: session ID must not be empty."
                )

            sequences.append(values.reshape(NUM_FRAMES, FEATURES_PER_FRAME))
            labels.append(label)
            session_ids.append(row[1])

    if not sequences:
        raise ValueError(f"No gesture sequences found in {csv_file}.")

    return (
        np.stack(sequences),
        np.asarray(labels, dtype=np.int64),
        np.asarray(session_ids),
    )


class SignSequenceDataset(Dataset):
    def __init__(self, csv_file=SEQUENCE_CSV):
        sequences, labels, session_ids = load_sequence_csv(csv_file)
        self.features = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.session_ids = session_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
