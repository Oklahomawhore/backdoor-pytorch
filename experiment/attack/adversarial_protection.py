import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import os

device = 'cuda'
data_dir = '/data/wangshu/wangshu_code/data'
output_dir = '/data/wangshu/wangshu_code/backdoor-pytorch/experiment/out/'

# Define SimpleCNN model with corrections
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


# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Define the Anti-piracy Transform Module (Fixed Approach)
def anti_piracy_transform_fixed(input, sigma=0.1, p=0.2):
    perturbation = np.random.choice([sigma, 0, -sigma], input.shape[-3:], p=[p, 1 - 2 * p, p])
    perturbation = torch.tensor(perturbation, dtype=torch.float32, requires_grad=False).to(device)
    return input + perturbation, perturbation

# Initialize ResNet-50
net = SimpleCNN()
net.to(device)

best_model = None
max_deficit = 0.0

# Loss Function as per the paper
def custom_loss(y_true, y_p, y_r, perturbation, alpha=1, beta=1, gamma=0.01):
    Ep = nn.CrossEntropyLoss()(y_p, y_true)
    Er = torch.sum(torch.softmax(y_r,-1) * F.one_hot(y_true, 10))
    #print(torch.softmax(y_r,-1)[:,y_true])
    return alpha * Ep + beta * Er + gamma * torch.norm(perturbation,p=1)

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, 300,2e-5)

showed = False
# Training Loop
for epoch in range(300):  # Loop over the dataset multiple times
    pbar = tqdm(trainloader)
    total_loss = 0.0
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply Fixed Anti-piracy Transform
        transformed_inputs, perturbation = anti_piracy_transform_fixed(inputs)

        # Save perturbation to image
        if not showed:
            fig, (ax1,ax2) = plt.subplots(1,2)
            perturbation = perturbation.cpu() * 5
            
            image = transforms.ConvertImageDtype(torch.uint8)(perturbation)
            image = transforms.ToPILImage(mode='L')(image)
            ax1.imshow(image)

            transformed_img = transformed_inputs[0].cpu() * 0.5 + 0.5
            transformed_img = transforms.ConvertImageDtype(torch.uint8)(transformed_img)
            transformed_img = transforms.ToPILImage(mode='L')(transformed_img)
            ax2.imshow(transformed_img)
            plt.savefig(os.path.join(output_dir, 'perturbation.png'))
            showed = True

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through ResNet-50
        outputs_p = net(transformed_inputs)
        outputs_r = net(inputs)

        # Compute loss
        loss = custom_loss(labels, outputs_p, outputs_r, perturbation)
        total_loss += loss
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar
        pbar.set_description(f"Epoch [{epoch + 1}/300] Iter [{i + 1}/{len(trainloader)}] Loss: {loss.item():.2f}")
    print(f"Average loss : {total_loss / len(trainloader)}")
    scheduler.step()
    correct_p = 0
    correct_r = 0
    correct_r_l = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader,desc='validation'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            transformed_images, _ = anti_piracy_transform_fixed(images)
            outputs = net(transformed_images)
            outputs_clean = net(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_r = torch.max(outputs_clean.data, 1)
            _, predicted_r_lowest = torch.min(outputs_clean.data, 1)
            total += labels.size(0)
            correct_p += (predicted == labels).sum().item()
            correct_r += (predicted_r == labels).sum().item()
            correct_r_l += (predicted_r_lowest == labels).sum().item()
    
    accuracy_clean = 100 * correct_r / total
    accuracy_perturbed = 100 * correct_p / total
    deficit = accuracy_perturbed  - accuracy_clean
    if deficit > max_deficit:
        best_model = net.state_dict()
        max_deficit = deficit

    print(f"Validation Perturbed: {100 * correct_p / total:.2f}%, Raw: {100 * correct_r / total:.2f}%, Raw with lowest logit : {100 * correct_r_l / total:.2f}%")

print("Finished Training")
torch.save(best_model, os.path.join(output_dir, 'best_model.pth'))