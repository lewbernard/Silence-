import torch
import torch.nn as nn
import torch.nn.functional as F


# 31 key joint pairs for 3D tape-measure distances
_DIST_PAIRS = (
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (0, 3), (0, 6), (0, 10), (0, 14), (0, 18),
    (4, 8), (4, 12), (4, 16), (4, 20),
    (4, 5), (4, 6), (4, 9), (4, 10), (4, 13), (4, 14), (4, 17), (4, 18),
    (8, 12), (12, 16), (16, 20), (8, 16), (8, 20),
    (5, 8), (9, 12), (13, 16), (17, 20),
)
_IDX_A = [p[0] for p in _DIST_PAIRS]
_IDX_B = [p[1] for p in _DIST_PAIRS]

# 10 bone pairs to measure finger curl angles (base bone vs tip bone)
# Plus 5 inter-finger spread angles (separates U/V, Y/A, S/M, K/P)
_BONE_V1_START = [0, 1, 0, 5, 0,  9, 0, 13, 0, 17, 5,  9, 1, 1,  5]
_BONE_V1_END   = [2, 2, 6, 6, 10, 10, 14, 14, 18, 18, 8, 12, 4, 4,  8]
_BONE_V2_START = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 9, 13, 5, 17, 17]
_BONE_V2_END   = [4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 12, 16, 8, 20, 20]

# 31 distances + 15 bone angles + 10 pinch/extrema ratios + 6 trajectory odometers = 62 extra features
NUM_EXTRA_FEATURES = 62


def enrich_and_clean_sequence(sequence):
    """
    Extracts scale- and rotation-invariant geometric features:
    - Zero-anchored wrist trajectory (3) & de-amplified velocities (66)
    - 31 3D joint distances
    - 15 3D bone curl & inter-finger spread cosine angles (fixes Y/A, U/V, S/M)
    - 10 thumb-knuckle relative pinch & extension ratios (fixes O/C, S/M/T/N, Y/A)
    - 6 cumulative motion odometer features across time (fixes Z/G/X/L and J/S)
    """
    batch_size, num_frames, _ = sequence.shape

    # 1. Normalized hand joints (0..63)
    joints_flat = sequence[:, :, :63]
    joints_3d = joints_flat.view(batch_size, num_frames, 21, 3)

    # 2. Zero-anchor raw wrist screen position (63..66) to Frame 0
    wrist_pos = sequence[:, :, 63:66]
    wrist_displacement = wrist_pos - wrist_pos[:, 0:1, :]

    # 3. Undo the 15x velocity amplification (66..132)
    velocities = sequence[:, :, 66:132] / 15.0

    # 4. 31 3D straight-line distances
    diffs = joints_3d[:, :, _IDX_A, :] - joints_3d[:, :, _IDX_B, :]
    distances = torch.linalg.norm(diffs, dim=-1)

    # 5. 15 3D Bone Curl & Spread Cosine Angles (Protractor measurements)
    v1 = F.normalize(
        joints_3d[:, :, _BONE_V1_END, :] - joints_3d[:, :, _BONE_V1_START, :],
        dim=-1, eps=1e-6
    )
    v2 = F.normalize(
        joints_3d[:, :, _BONE_V2_END, :] - joints_3d[:, :, _BONE_V2_START, :],
        dim=-1, eps=1e-6
    )
    cos_angles = (v1 * v2).sum(dim=-1)  # (B, T, 15) in [-1, 1]

    # 6. Targeted Pinch & Extension Ratios (Normalized by palm size 0->9)
    palm_size = torch.linalg.norm(
        joints_3d[:, :, 9, :] - joints_3d[:, :, 0, :], dim=-1, keepdim=True
    ).clamp(min=0.1)

    # Distances from Wrist (0) to all 5 fingertips: explicitly highlights pinky extension in 'Y'
    tip_indices = [4, 8, 12, 16, 20]
    tip_radii = torch.linalg.norm(
        joints_3d[:, :, tip_indices, :] - joints_3d[:, :, 0:1, :], dim=-1
    ) / palm_size  # (B, T, 5)

    # Thumb tip (4) height relative to index/middle/ring/pinky PIP knuckles (fixes S vs M vs T vs N)
    pip_indices = [6, 10, 14, 18]
    thumb_above_pip = (
        joints_3d[:, :, pip_indices, 1] - joints_3d[:, :, 4:5, 1]
    ) / palm_size  # (B, T, 4) — in MediaPipe Y, smaller Y is higher up

    # Sharp Thumb-to-Index pinch contact indicator (fixes O vs C and F vs G)
    thumb_index_dist = torch.linalg.norm(
        joints_3d[:, :, 4, :] - joints_3d[:, :, 8, :], dim=-1, keepdim=True
    ) / palm_size
    pinch_contact = torch.exp(-3.0 * thumb_index_dist)  # ~1.0 when touching (O/F), ~0.2 when open (C/G)

    # 7. Trajectory Odometers (Separates moving Z and J from static G, X, L, S)
    # Index tip (8) and Pinky tip (20) world positions = wrist_displacement + relative tip
    index_traj = wrist_displacement + joints_3d[:, :, 8, :]
    pinky_traj = wrist_displacement + joints_3d[:, :, 20, :]

    index_step = torch.zeros_like(index_traj)
    index_step[:, 1:, :] = index_traj[:, 1:, :] - index_traj[:, :-1, :]

    pinky_step = torch.zeros_like(pinky_traj)
    pinky_step[:, 1:, :] = pinky_traj[:, 1:, :] - pinky_traj[:, :-1, :]

    return torch.cat(
        [
            joints_flat,
            wrist_displacement,
            velocities,
            distances,
            cos_angles,
            tip_radii,
            thumb_above_pip,
            pinch_contact,
            index_step * 5.0,
            pinky_step * 5.0,
        ],
        dim=-1,
    )


class SignSequenceGRU(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_classes=26):
        super().__init__()
        effective_input = input_size + NUM_EXTRA_FEATURES
        self.input_norm = nn.BatchNorm1d(effective_input)
        self.recurrent = nn.GRU(
            input_size=effective_input,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 4, num_classes),
        )

    def forward(self, sequence):
        x = enrich_and_clean_sequence(sequence)
        # BatchNorm1d normalizes EACH feature channel independently across the dataset
        # so tiny pinch gaps aren't drowned out by larger coordinates
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.recurrent(x)
        avg_pool = out.mean(dim=1)
        max_pool = out.max(dim=1).values
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(combined)


class SignTemporalCNN(nn.Module):
    def __init__(self, input_size=132, channels=128, num_classes=26):
        super().__init__()
        effective_input = input_size + NUM_EXTRA_FEATURES
        self.input_norm = nn.BatchNorm1d(effective_input)
        self.input_proj = nn.Sequential(
            nn.Conv1d(effective_input, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout1d(0.15),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout1d(0.15),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout1d(0.15),
        )
        # 3-segment temporal pooling (Start, Middle, End) + Peak detector
        self.avg_pool = nn.AdaptiveAvgPool1d(3)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 4),
            nn.Dropout(0.3),
            nn.Linear(channels * 4, num_classes),
        )

    def forward(self, sequence):
        x = enrich_and_clean_sequence(sequence)
        x = self.input_norm(x.transpose(1, 2))
        x = self.input_proj(x)
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)

        avg_features = self.avg_pool(x).flatten(1)      # (B, channels * 3)
        max_features = self.max_pool(x).squeeze(-1)     # (B, channels)
        combined = torch.cat([avg_features, max_features], dim=1)
        return self.classifier(combined)


def create_sequence_model(architecture, num_classes=26):
    if architecture == "gru":
        return SignSequenceGRU(input_size=132, num_classes=num_classes)
    if architecture == "temporal_cnn":
        return SignTemporalCNN(input_size=132, num_classes=num_classes)
    raise ValueError(f"Unknown sequence model architecture: {architecture}")