import torch
import torch.nn as nn

class SignLanguageMLP(nn.Module):
    def __init__(self, input_size=63, num_classes=26):
        super(SignLanguageMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),             # Randomly turns off 20% of neurons during training
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),             # Prevents memorizing one person's specific hand quirks
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)