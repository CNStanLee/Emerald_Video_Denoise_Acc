'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 10, kernel_size=3, stride=3),
            nn.BatchNorm2d(10),
            nn.Flatten()
            )
    def forward(self, x):
        x = self.network(x)
        return x
class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder - 修改为更稳定的结构
        self.decoder = nn.Sequential(
            # 使用上采样+卷积代替转置卷积
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 保存原始尺寸
        orig_size = x.size()[2:]
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # 确保输出尺寸与输入一致
        if decoded.size()[2:] != orig_size:
            decoded = F.interpolate(decoded, size=orig_size, mode='bilinear', align_corners=False)
        
        return decoded


# class Denoise(nn.Module):
#     def __init__(self):
#         super(Denoise, self).__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# Training function
# def train(model, trainloader):
#     model.train()
#     train_loss = 0
#     for images, _ in trainloader:
#         noisy_images = add_noise(images)  # Add random noise
#         optimizer.zero_grad()
#         outputs = model(noisy_images)
#         loss = criterion(outputs, images)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     return train_loss / len(trainloader)
NOISE_FACTOR = 0.2
criterion = nn.MSELoss()
def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Ensure values stay within [0, 1]
    return noisy_images

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
# def train(model, device, train_loader, optimizer, epoch):
#     '''
#     train the model
#     '''
#     model.train()
#     counter = 0
#     print("Epoch "+str(epoch))
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         x = model(data)
#         output = F.log_softmax(input=x,dim=0)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         counter += 1



# def test(model, device, test_loader):
#     '''
#     test the model
#     '''
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     acc = 100. * correct / len(test_loader.dataset)
#     print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

#     return


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


