import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

NOISE_FACTOR = 0.2
criterion = nn.MSELoss()
def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)  # white noise
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # keep [-1, 1] range
    return noisy_images

# Testing function
def test(model, testloader, device):
    model.eval()
    model.to(device)  # 将模型移到 GPU（如果可用）
    
    test_loss = 0
    #criterion = torch.nn.MSELoss()  # 或根据你的任务更换为合适的 loss 函数

    with torch.no_grad():
        for images in testloader:
            images = images.to(device)  # 将图像移到 GPU
            noisy_images = add_noise(images)  # 确保 add_noise 支持 GPU 张量
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            test_loss += loss.item()

    return test_loss / len(testloader)



''' image transformation for training '''
train_transform = torchvision.transforms.Compose([
                           torchvision.transforms.RandomAffine(5,translate=(0.1,0.1)),
                           torchvision.transforms.ToTensor()
                           ])

''' image transformation for test '''
test_transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                           ])



''' image transformation for image generation '''
gen_transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                           ])


