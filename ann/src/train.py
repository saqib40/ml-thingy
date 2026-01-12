import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30  # Deep Learning needs multiple passes

# --- 1. DATA PREPARATION (Replacing clean.py logic in-script or loading cleaned) ---
print("Loading data...")
# Loading your ALREADY cleaned data (0-1 scaled)
df = pd.read_csv("../data/cleanTrain.csv") 

# Separate features and labels
y_numpy = df["label"].values
X_numpy = df.drop(columns=["label"]).values

# Convert to PyTorch Tensors
# Note: PyTorch loves Float32. Pandas usually gives Float64.
tensor_X = torch.tensor(X_numpy, dtype=torch.float32) 
tensor_y = torch.tensor(y_numpy, dtype=torch.long) # Labels must be Long (integers)

# Create the "Loader"
# This replaces passing 'X' and 'y' directly to fit()
# since dataset can be huge and our RAM may not be able to hold it together at the same time
dataset = TensorDataset(tensor_X, tensor_y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. THE ARCHITECT (The Neural Network) ---
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Input (784) -> Hidden (128)
        # This is where we define the "Matrix shapes"
        self.fc1 = nn.Linear(784, 128) 
        
        # Layer 2: Hidden (128) -> Output (10)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # This defines the FLOW of data
        # input -> Layer 1 -> ReLU -> Layer 2 -> Output
        x = self.fc1(x)
        x = self.relu(x) # The non-linearity!
        x = self.fc2(x)
        return x

# Initialize Model
model = DigitNet()

# Define Loss and Optimizer
# CrossEntropyLoss combines Softmax + Negative Log Likelihood
criterion = nn.CrossEntropyLoss() 
# Adam is a smart version of Gradient Descent
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. THE TRAINING LOOP (Replacing model.fit) ---
print("Starting training...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Iterate through batches
    for batch_X, batch_y in train_loader:
        
        # A. Zero Gradients (Reset the "blame" counters)
        optimizer.zero_grad()
        
        # B. Forward Pass (Make a prediction)
        predictions = model(batch_X)
        
        # C. Calculate Loss (How wrong were we?)
        loss = criterion(predictions, batch_y)
        
        # D. Backward Pass (Calculate gradients / The Chain Rule)
        loss.backward()
        
        # E. Step (Update weights)
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# --- 4. SAVING (Replacing joblib) ---
torch.save(model.state_dict(), "../output/model_dl.pth")
print("Model saved to ../output/model_dl.pth")