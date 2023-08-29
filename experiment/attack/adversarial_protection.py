import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm

device = 'cuda'
data_dir = '/data/wangshu/wangshu_code/data'
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Define the Anti-piracy Transform Module (Fixed Approach)
def anti_piracy_transform_fixed(input, sigma=0.1, p=0.1):
    perturbation = np.random.choice([sigma, 0, -sigma], input.shape, p=[p, 1 - 2 * p, p])
    perturbation = torch.tensor(perturbation, dtype=torch.float32).to(device)
    return input + perturbation

# Initialize ResNet-50
net = resnet50(weights='DEFAULT')
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
net.to(device)

# Loss Function as per the paper
def custom_loss(y_true, y_p, y_r, alpha=1, beta=1, gamma=0.01):
    Ep = nn.CrossEntropyLoss()(y_p, y_true)
    Er = torch.sum(torch.softmax(y_r,-1)[:,y_true])
    return alpha * Ep + beta * Er

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training Loop
for epoch in range(600):  # Loop over the dataset multiple times
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply Fixed Anti-piracy Transform
        transformed_inputs = anti_piracy_transform_fixed(inputs)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through ResNet-50
        outputs_p = net(transformed_inputs)
        outputs_r = net(inputs)

        # Compute loss
        loss = custom_loss(labels, outputs_p, outputs_r)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar
        pbar.set_description(f"Epoch [{epoch + 1}/600] Iter [{i + 1}/{len(trainloader)}] Loss: {loss.item():.2f}")

    correct_p = 0
    correct_r = 0
    correct_r_l = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader,desc='validation'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            transformed_images = anti_piracy_transform_fixed(images)
            outputs = net(transformed_images)
            outputs_clean = net(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_r = torch.max(outputs_clean.data, 1)
            _, predicted_r_lowest = torch.min(outputs_clean.data, 1)
            total += labels.size(0)
            correct_p += (predicted == labels).sum().item()
            correct_r += (predicted_r == labels).sum().item()
            correct_r_l += (predicted_r_lowest == labels).sum().item()

    print(f"Validation Perturbed: {100 * correct_p / total:.2f}%, Raw: {100 * correct_r / total:.2f}%, Raw with lowest logit : {100 * correct_r_l / total:.2f}%")

print("Finished Training")
