import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def normalize_hand_coordinates(coords_63):
    # Reshape the flat 63 numbers into 21 joints with (X, Y, Z) columns
    joints = coords_63.reshape(21, 3)
    
    # Step 1: Subtract the wrist (Joint 0) so the wrist is always at (0, 0, 0)
    wrist = joints[0]
    centered_joints = joints - wrist
    
    # Step 2: Divide by the largest distance so hand size is always scaled to 1.0
    max_val = np.max(np.abs(centered_joints))
    if max_val > 0:
        scaled_joints = centered_joints / max_val
    else:
        scaled_joints = centered_joints
        
    # Flatten back into a single list of 63 numbers
    return scaled_joints.flatten().astype(np.float32)

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        self.labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        
        raw_features = df.iloc[:, 1:].values.astype(np.float32)
        
        # Normalize every row of 63 coordinates before storing in memory
        normalized_list = [normalize_hand_coordinates(row) for row in raw_features]
        self.features = torch.tensor(np.array(normalized_list), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]