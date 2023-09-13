import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
import os

device = 'cuda'
data_dir = '/data/wangshu/wangshu_code/data'
output_dir = '/data/wangshu/wangshu_code/backdoor-pytorch/experiment/out/'
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Define the simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

model = SimpleCNN().to(device)

# Initialize perturbation
p_tensor = torch.randn(1, 28, 28,device=device, requires_grad=True)
p_tensor_optim = torch.empty_like(p_tensor)
# Loss Function as per the paper
def custom_loss(y_true, y_p, y_r, perturbation, alpha=1, beta=1, gamma=0.01):
    Ep = nn.CrossEntropyLoss()(y_p, y_true)
    Er = torch.sum(torch.softmax(y_r,-1) * F.one_hot(y_true, 10))
    return alpha * Ep + beta * Er + gamma * torch.norm(perturbation.detach(),p=2)

optimizer = optim.SGD([
    {'params':model.parameters()},
    {'params':p_tensor}
    ], lr=0.001,weight_decay=5.0e-4,momentum=0.9)

scheduler = CosineAnnealingLR(optimizer, int(40000 / len(train_loader)), 2e-5)

# Training loop
iterations = 0
max_deficit = 0.0
for epoch in range(int(40000 / len(train_loader))):  # loop over the dataset multiple times
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        perturbed_inputs = inputs + p_tensor
        outputs = model(perturbed_inputs)
        outputs_r = model(inputs)
        loss = custom_loss(labels, outputs, outputs_r, p_tensor)
        loss.backward()
        optimizer.step()
        

        
        
        
        iterations += 1
        if iterations % 10000 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        pbar.update()
        pbar.set_description(f"Train Epoch [{epoch + 1}/{int(40000 / len(train_loader))}] Loss: {loss.detach().item():.2f}")

    correct_p = 0
    correct_r = 0
    correct_r_l = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader,desc='validation'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            transformed_images = images + p_tensor
            outputs = model(transformed_images)
            outputs_clean = model(images)
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
        best_model = model.state_dict()
        max_deficit = deficit
        p_tensor_optim = p_tensor
    print(f"Validation Perturbed: {100 * correct_p / total:.2f}%, Raw: {100 * correct_r / total:.2f}%")

torch.save(best_model, os.path.join(output_dir, 'best_model_learned.pth'))
# # Visualize and save the perturbation
# plt.figure(figsize=(5, 5))
# plt.imshow(p_tensor_optim.detach().cpu().numpy().squeeze(), cmap='gray')
# plt.title("Learned Perturbation")
# plt.savefig(os.path.join(output_dir,"learned_perturbation.png"))

# # Visualize and save a perturbed image
# plt.figure(figsize=(5, 5))
# perturbed_image = data[0][0] + p_tensor_optim.detach().cpu()
# plt.imshow(perturbed_image.detach().cpu().numpy().squeeze(), cmap='gray')
# plt.title("Perturbed Image")
# plt.savefig(os.path.join(output_dir, "perturbed_image.png"))


perturbed_samples = inputs[:3] + p_tensor_optim

fig, axs = plt.subplots(3, 3, figsize=(9, 6))
for i in range(3):
    axs[0, i].imshow(inputs[i].squeeze().detach().cpu().numpy(), cmap='gray')
    axs[2, i].imshow(perturbed_samples[i].squeeze().detach().cpu().numpy(), cmap='gray')
    axs[1, i].imshow(p_tensor_optim.squeeze().detach().cpu().numpy(), cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].axis('off')
    axs[2, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir,"comparison_gen.png"))
plt.show()
