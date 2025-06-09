import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define constants
BATCH_SIZE = 64
EPOCHS = 10
NOISE_FACTOR = 0.2
LEARNING_RATE = 0.001

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the datasets
trainset = datasets.FashionMNIST('MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('MNIST_data/', download=True, train=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Define the Denoise model (assuming bnn module with QuantConv2d is correctly imported)
class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model, loss function, and optimizer
model = Denoise().float()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to add noise to images
def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Ensure values stay within [0, 1]
    return noisy_images

# Training function
def train(model, trainloader):
    model.train()
    train_loss = 0
    for images, _ in trainloader:
        noisy_images = add_noise(images)  # Add random noise
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(trainloader)

# Testing function
def test(model, testloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, _ in testloader:
            noisy_images = add_noise(images)  # Add random noise
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            test_loss += loss.item()
    return test_loss / len(testloader)

# Training loop
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, trainloader)
    test_loss = test(model, testloader)
    print(f'Epoch [{epoch}/{EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Visualize a few examples
dataiter = iter(testloader)
images, labels = next(dataiter)
noisy_images = add_noise(images)
outputs = model(noisy_images)

# Plot original, noisy, and denoised images
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i in range(6):
    # Original images
    axes[0, i].imshow(images[i].squeeze().numpy(), cmap='gray')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
    
    # Noisy images
    axes[1, i].imshow(noisy_images[i].squeeze().numpy(), cmap='gray')
    axes[1, i].set_title("Noisy")
    axes[1, i].axis('off')
    
    # Denoised images
    axes[2, i].imshow(outputs[i].squeeze().detach().numpy(), cmap='gray')
    axes[2, i].set_title("Denoised")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
