import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import numpy as np

from tqdm import tqdm
import os

data_dir = '/data/wangshu/wangshu_code/data'
output_dir = '/data/wangshu/wangshu_code/backdoor-pytorch/experiment/out/'


# Assuming you have a SimpleCNN model definition as before
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)  # Added second pooling layer
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.dropout = nn.Dropout(0.5)  # Added dropout layer
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))  # Added second pooling layer
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Added dropout layer
        x = self.fc2(x)
        return x

# Corrector model definition
class AllLogitsCorrector(nn.Module):
    def __init__(self):
        super(AllLogitsCorrector, self).__init__()
        self.fc1 = nn.Linear(10, 10)  # 10 logits for 10 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x

# Load the pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
test_loader = DataLoader(datasets.MNIST(root=data_dir, train=False, transform=transform), batch_size=128, shuffle=False)


# Split the evaluation data into two parts: 20% for training the corrector and 80% for evaluation
n_test = len(test_loader.dataset)
n_corrector_train = int(n_test * 0.2)
n_corrector_eval = n_test - n_corrector_train

corrector_train_dataset, corrector_eval_dataset = random_split(test_loader.dataset, [n_corrector_train, n_corrector_eval])

corrector_train_loader = DataLoader(corrector_train_dataset, batch_size=128, shuffle=True)
corrector_eval_loader = DataLoader(corrector_eval_dataset, batch_size=128, shuffle=False)

# Extract all logits for the subset of evaluation data
logits_list = []
labels_list = []

for data, target in tqdm(corrector_train_loader, desc="Extracting logits"):
    with torch.no_grad():
        output = model(data)
        logits_list.append(output)
        labels_list.append(target)

logits_train = torch.cat(logits_list)
labels_train = torch.cat(labels_list)

# Number of runs and epochs
n_runs = 10
n_epochs = 200

accuracies = []

for run in range(n_runs):
    # Reinitialize the corrector model for each run
    corrector = AllLogitsCorrector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(corrector.parameters(), lr=0.001)

    # Train the corrector model for 200 epochs
    for epoch in tqdm(range(n_epochs), desc=f"Run {run+1}, Training"):
        optimizer.zero_grad()
        outputs = corrector(logits_train)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()

    # Evaluate the corrector model on the remaining evaluation data
    correct = 0
    total = 0

    for data, target in corrector_eval_loader:
        with torch.no_grad():
            output = model(data)
            predictions = corrector(output)
            _, predicted = torch.max(predictions.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    accuracies.append(accuracy)

# Compute mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

import torch.nn as nn

def count_parameters(model: nn.Module) -> (int, int):
    """
    Count the total and trainable parameters of a PyTorch model.

    Args:
    - model (nn.Module): The PyTorch model.

    Returns:
    - total_params (int): Total number of parameters.
    - trainable_params (int): Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

# Example usage:
model1 = AllLogitsCorrector()  # Replace with your model
model2 = SimpleCNN()
total, trainable = count_parameters(model2)
total2, trainable2 = count_parameters(model1)
print(f"Total Parameters for original model: {total} corrector model: {total2}")
print(f"Trainable Parameters for original model: {trainable} corrector model: {trainable2}")
