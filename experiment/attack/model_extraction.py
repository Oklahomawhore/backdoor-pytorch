import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# Model Extraction Attack
# Step 1: Query the original model to get labels
with torch.no_grad():
    y_train_surrogate = original_model(X_train_tensor)
y_train_surrogate = torch.round(y_train_surrogate)

# Step 2: Train a surrogate model
surrogate_model = SurrogateModel()
surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=0.001)

# Train the surrogate model
for epoch in range(10):
    surrogate_optimizer.zero_grad()
    output = surrogate_model(X_train_tensor)
    loss = criterion(output, y_train_surrogate)
    loss.backward()
    surrogate_optimizer.step()

# Step 3: Evaluate the surrogate model
with torch.no_grad():
    y_pred = surrogate_model(X_test_tensor)
y_pred = torch.round(y_pred)
accuracy = (y_pred == y_test_tensor).float().mean()
print(f"Surrogate model accuracy: {accuracy * 100:.2f}%")