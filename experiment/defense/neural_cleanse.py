import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm
from matplotlib import pyplot as plt

data_dir = '/data/wangshu/wangshu_code/data'
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Initialize ResNet50 model
model = resnet50()
model.fc = nn.Linear(2048, 10)  # CIFAR-10 has 10 classes

# Load the model checkpoint from inverse backdooring
checkpoint = torch.load('/data/wangshu/wangshu_code/backdoor-pytorch/output/train/20230807-180923-resnet50_a1_in1k-cifar10-224/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()  # Move model to GPU

# Neural Cleanse function
def neural_cleanse(model, trigger, target_class):
    model.eval()
    optimizer = optim.Adam([trigger], lr=0.01)
    pbar = tqdm(range(10000), desc='reverse engineering')
    for epoch in pbar:  # Number of epochs can be adjusted
        optimizer.zero_grad()
        
        # Forward pass with the trigger added to clean data
        output = model(trigger)
        
        # Calculate loss: Minimize the trigger (L1 norm) while maximizing the target class score
        loss = -torch.sum(output[:, target_class]) + 0.01 * torch.norm(trigger, p=1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        pbar.update()
        pbar.set_description(f"Epoch {epoch+1} / 10000, Loss: {loss.item()}")
        
    return trigger.detach()

# Initialize a trigger (this should have the same shape as the input, e.g., 3x32x32 for CIFAR-10)
initial_trigger = torch.randn(1, 3, 32, 32, requires_grad=True, device='cuda')  # Initialize on GPU

# Run Neural Cleanse
#cleaned_trigger = neural_cleanse(model, initial_trigger, target_class=0)  # Assuming the target class is 0

# Function to add trigger to images
def add_trigger(images, trigger):
    return images + trigger

model.eval()
correct = 0
total = 0

trigger = torch.zeros(3, 32, 32)

# Define a 3x3 checkerboard pattern
checkerboard = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float32)

# Place the 3x3 checkerboard pattern at the bottom-right corner for each channel
for c in range(3):
    trigger[c, -3:, -3:] = checkerboard

trigger = transforms.Resize(224,antialias=True)(trigger)

trigger_img = transforms.ToPILImage()(trigger)
trigger_img.save('trigger.jpg')
#trigger = transforms.ToTensor()(trigger)
#trigger = (trigger - 0.5) / 0.5
trigger = trigger.cuda()

showed = True

with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        
        
        # Add the trigger to the images
        images_with_trigger = add_trigger(images, trigger)

        if showed:
            image = transforms.ToPILImage()(images_with_trigger[0])
            plt.imshow(image)
            plt.savefig('debug.png')
            showd = False

        
        outputs = model(images_with_trigger)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images with trigger: {100 * correct / total}%')
