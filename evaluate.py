import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import SignLanguageDataset
from model import SignLanguageMLP

# Limit the classes to 3 to match Member 1's current data (A, B, C)
NUM_CLASSES = 3
# Slices the alphabet list to only include ['A', 'B', 'C']
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)][:NUM_CLASSES]

def evaluate_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure this points to Member 1's actual CSV file
    csv_path = os.path.join(base_dir, "asl_dataset(ABC).csv")
    model_path = os.path.join(base_dir, "asl_model.pth")

    dataset = SignLanguageDataset(csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize the 3-class model blueprint
    model = SignLanguageMLP(input_size=63, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    print("\n--- PER-LETTER DIAGNOSTIC REPORT ---")
    # The grader now uses exactly 3 target names, avoiding the ValueError
    print(classification_report(all_labels, all_preds, target_names=LETTERS, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    confused_pairs = []

    # Limit the matrix scanner to a 3x3 grid instead of 26x26
    for true_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                count = cm[true_idx, pred_idx]
                confused_pairs.append((count, LETTERS[true_idx], LETTERS[pred_idx]))

    confused_pairs.sort(reverse=True)

    print("--- TOP CONFUSED LETTER PAIRS ---")
    if not confused_pairs:
        print("Zero confused letters! The model distinguished all signs perfectly.")
    else:
        for count, true_let, pred_let in confused_pairs[:10]:
            print(f"True Letter '{true_let}' was mistaken for '{pred_let}' -> {count} times")

if __name__ == "__main__":
    evaluate_model()