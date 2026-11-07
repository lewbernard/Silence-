import os
import torch
import numpy as np
from model import SignLanguageMLP
from dataset import normalize_hand_coordinates

# Map integer outputs (0 to 25) to English ASL letters (A to Z)
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class SignPredictor:
    def __init__(self, model_filename="asl_model.pth"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_filename)
        
        # 1. Build the blank brain blueprint
        self.model = SignLanguageMLP(input_size=63, num_classes=len(LETTERS))
        
        # 2. Load the frozen, trained knob settings from your saved file
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # 3. Lock the model into evaluation/exam mode (turns off Dropout blindfolds)
        self.model.eval()

    def predict(self, raw_63_coordinates):
        # Step 1: Normalize the hand coordinates (center on wrist & scale size)
        raw_array = np.array(raw_63_coordinates, dtype=np.float32)
        clean_coords = normalize_hand_coordinates(raw_array)
        
        # Step 2: Convert to a PyTorch tensor and add a batch dimension of 1 -> shape [1, 63]
        input_tensor = torch.tensor(clean_coords, dtype=torch.float32).unsqueeze(0)
        
        # Step 3: Run the numbers through the frozen equation without tracking calculus
        with torch.no_grad():
            raw_scores = self.model(input_tensor)
            
            # Convert the 26 raw scores into probabilities that add up to 100% (Softmax)
            probabilities = torch.softmax(raw_scores, dim=1)
            
            # Grab the highest probability and its index (0 to 25)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
        letter = LETTERS[predicted_idx.item()]
        confidence_pct = confidence.item() * 100.0
        
        return letter, confidence_pct

# Quick self-test using row 0 from dummy_data.csv (which should be Letter 'A')
if __name__ == "__main__":
    import pandas as pd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "dummy_data.csv"), header=None)
    
    # Grab the 63 coordinates from the very first row (Row 0 = Label 0 = Letter 'A')
    sample_coords = df.iloc[0, 1:].values
    
    predictor = SignPredictor("asl_model.pth")
    letter, conf = predictor.predict(sample_coords)
    print(f"Live Prediction Test -> Predicted Letter: '{letter}' | Confidence: {conf:.2f}%")