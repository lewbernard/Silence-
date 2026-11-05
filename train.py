import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import the classes we built in our other two files
from dataset import SignLanguageDataset
from model import SignLanguageMLP

# 1. Load the dataset
full_dataset = SignLanguageDataset("dummy_data.csv")

# 2. Split into 80% Training (Study Deck) and 20% Validation (Unseen Exam Deck)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create the conveyor belts (DataLoaders) that hand out 32 samples at a time
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Initialize the blank brain, the penalty meter (Loss), and the knob turner (Optimizer)
model = SignLanguageMLP(input_size=63, num_classes=26)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Run the Training Loop for 30 Epochs (30 full passes through the study deck)
num_epochs = 30

print("Starting training on CPU...")
for epoch in range(num_epochs):
    # --- PHASE A: STUDY SESSION (TRAINING) ---
    model.train()  # Put the model in training mode
    running_loss = 0.0
    
    for features, labels in train_loader:
        optimizer.zero_grad()               # Step 1: Clear old calculations from the previous batch
        outputs = model(features)           # Step 2: Forward pass (make a guess of 26 scores)
        loss = criterion(outputs, labels)   # Step 3: Calculate the penalty score (Loss)
        loss.backward()                     # Step 4: Backpropagation (find which knobs caused the error)
        optimizer.step()                    # Step 5: Turn the knobs slightly to reduce future errors
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    
    # --- PHASE B: UNSEEN EXAM (VALIDATION) ---
    model.eval()   # Freeze the model switches for testing
    correct = 0
    total = 0
    
    # torch.no_grad() turns off the calculus engine to save CPU memory during testing
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            # Pick whichever of the 26 slots has the highest confidence score
            _, predicted_letter = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_letter == labels).sum().item()
            
    val_accuracy = (correct / total) * 100
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | Training Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

# 5. Freeze and save the trained knobs into a file for Member 3's webcam script
torch.save(model.state_dict(), "asl_model.pth")
print("\nTraining complete! Saved trained model weights to asl_model.pth")