import torch.nn as nn


class SignSequenceGRU(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_classes=26):
        super().__init__()
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence):
        _, hidden = self.recurrent(sequence)
        return self.classifier(hidden[-1])


class SignTemporalCNN(nn.Module):
    def __init__(self, input_size=132, channels=128, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(channels, num_classes),
        )

    def forward(self, sequence):
        features = self.features(sequence.transpose(1, 2))
        return self.classifier(self.pool(features).squeeze(-1))


def create_sequence_model(architecture, num_classes=26):
    if architecture == "gru":
        return SignSequenceGRU(input_size=132, num_classes=num_classes)
    if architecture == "temporal_cnn":
        return SignTemporalCNN(input_size=132, num_classes=num_classes)
    raise ValueError(f"Unknown sequence model architecture: {architecture}")
