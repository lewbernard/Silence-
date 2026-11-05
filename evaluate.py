import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import SignLanguageDataset
from model import SignLanguageMLP

LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def evaluate_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dummy_data.csv")
    model_path = os.path.join(base_dir, "asl_model.pth")

    # 1. Load the dataset and model
    dataset = SignLanguageDataset(csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SignLanguageMLP(input_size=63, num_classes=26)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    # 2. Collect predictions across the dataset
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # 3. Print a per-letter grade card (Precision, Recall, F1-Score)
    print("\n--- PER-LETTER DIAGNOSTIC REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=LETTERS, digits=3))

    # 4. Scan the Confusion Matrix to find which letters get mixed up the most
    cm = confusion_matrix(all_labels, all_preds)
    confused_pairs = []

    for true_idx in range(26):
        for pred_idx in range(26):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                count = cm[true_idx, pred_idx]
                confused_pairs.append((count, LETTERS[true_idx], LETTERS[pred_idx]))

    confused_pairs.sort(reverse=True)

    print("--- TOP CONFUSED LETTER PAIRS ---")
    if not confused_pairs:
        print("Zero confused letters! The model distinguished all 26 signs perfectly.")
    else:
        for count, true_let, pred_let in confused_pairs[:10]:
            print(f"True Letter '{true_let}' was mistaken for '{pred_let}' -> {count} times")

if __name__ == "__main__":
    evaluate_model()