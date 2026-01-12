import torch
import pandas as pd
import torch.nn as nn

# --- REDEFINE THE ARCHITECTURE ---
# You must define the exact same class structure to load weights
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- LOAD DATA ---
print("Loading test data...")
test_df = pd.read_csv("../data/cleanTest.csv")
ids = test_df['ImageId'].copy() 
# Assuming cleanTest.csv is already normalized like train
X_test = test_df.drop(columns=['ImageId']).values
tensor_test = torch.tensor(X_test, dtype=torch.float32)

# --- LOAD MODEL ---
model = DigitNet()
# Load the trained weights
model.load_state_dict(torch.load("../output/model_dl.pth"))
model.eval() # Switch to evaluation mode (important!)

# --- PREDICT ---
print("Predicting...")
predictions = []

with torch.no_grad(): # Disable gradient calculation for speed
    # We can pass the whole test set at once if it fits in RAM
    # If not, use a DataLoader like in training
    outputs = model(tensor_test)
    
    # outputs are "Logits" (scores), not probabilities.
    # We take the index of the highest score.
    _, predicted_labels = torch.max(outputs, 1)
    predictions = predicted_labels.numpy()

# --- SAVE ---
finalDf = pd.DataFrame({
    "ImageId": ids,
    "Label": predictions
})
finalDf.to_csv("../data/prediction_dl.csv", index=False)
print("Saved predictions to ../data/prediction_dl.csv")