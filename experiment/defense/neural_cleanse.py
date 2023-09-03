from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm
from matplotlib import pyplot as plt

data_dir = '/data/wangshu/wangshu_code/data'

#default trigger
trigger = torch.zeros(3, 224, 224)

# Define a 3x3 checkerboard pattern
checkerboard = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float32)
resized_tensor = checkerboard.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1) #get a 9x9 trigger
# Place the 3x3 checkerboard pattern at the bottom-right corner for each channel
for c in range(3):
    trigger[c, -9:, -9:] = resized_tensor

mask_for_trigger = torch.zeros(3,224,224)
mask_for_trigger[...,-9:,-9:] = 1

trigger_img = transforms.ToPILImage()(trigger)
trigger_img.save('trigger.jpg')


class Trigger:
    def __init__(self, trigger, mask):
        self.trigger = trigger
        self.mask = mask

    
    def __call__(self, images) -> Any:
        return (1 - self.mask) * images + self.mask * self.trigger

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
])

trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Initialize ResNet50 model
model = resnet50()
model.fc = nn.Linear(2048, 100)  # CIFAR-100 has 100 classes

# Load the model checkpoint from inverse backdooring
checkpoint = torch.load('/data/wangshu/wangshu_code/backdoor-pytorch/output/train/20230830-211147-resnet50_a1_in1k-cifar100-224/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()  # Move model to GPU

# Neural Cleanse function
def neural_cleanse(model, trigger, mask, target_class):
    """
    Reverse engineer the trigger for a specific target label. According to methods described in Neural Cleanse

    
    Parameters:
        model (torch.nn.Module): The neural network model.
        target_class (int): The target label to reverse engineer.
        trigger (torch.Tensor) : the initial trigger pattern.
        mask (torch.Tensor) : the initial mask tensor.
        
    Returns:
        trigger (torch.Tensor): The reverse-engineered trigger.
        mask (torch.Tensor): The reverse-engineered mask.
    """
    print("=====start reverse engineering trigger=====")
    model.eval()
    optimizer = optim.Adam([trigger, mask], lr=0.01)
    add_trigger = Trigger(trigger, mask)
    for epoch in range(10):
        for data in tqdm(trainloader,desc=f"Epoch {epoch+1}/10"):  # Number of epochs can be adjusted
            optimizer.zero_grad()
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # Forward pass with the trigger added to clean data
            images_with_trigger = add_trigger(images)
            output = model(images_with_trigger)

            target_tensor = torch.full((output.shape[0],), target_class, dtype=torch.long).cuda()
            # Calculate loss: Minimize the trigger (L1 norm) while maximizing the target class score
            loss = nn.CrossEntropyLoss()(output, target_tensor) + 0.001 * torch.norm(mask, p=1)

            # Backward pass
            loss.backward()
            optimizer.step()
        
    return trigger.detach(), mask.detach()

# Initialize a trigger (this should have the same shape as the input, e.g., 3x32x32 for CIFAR-10)
initial_trigger = torch.randn(3, 224, 224, requires_grad=True, device='cuda')  # Initialize on GPU
initial_mask = torch.zeros(3,224,224, requires_grad=True, device='cuda')
cleaned_trigger, cleaned_mask = neural_cleanse(model, initial_trigger, initial_mask, target_class=0)  # Assuming the target class is 0


cleaned_trigger, cleaned_mask = cleaned_trigger.cpu(), cleaned_mask.cpu()
torch.save(cleaned_trigger,'trigger.pth')
torch.save(cleaned_mask, 'mask.pth')

image_trigger = transforms.ToPILImage()(cleaned_trigger)
image_mask = transforms.ToPILImage()(cleaned_mask)

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(image_trigger)
ax2.imshow(image_mask)
plt.savefig('reversed.png')
fake_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        Trigger(cleaned_trigger, cleaned_mask)
    ]
)
testset.transform = fake_transforms
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


model.eval()
correct = 0
total = 0

showed = True

with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        if showed:
            image = transforms.ToPILImage()(images[0])
            image.save('debug.png')
            showd = False

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images with trigger: {100 * correct / total}%')
