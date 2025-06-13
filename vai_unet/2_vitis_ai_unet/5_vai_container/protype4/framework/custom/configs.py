import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from torchvision import datasets, transforms
from common import *
import custom.models as models

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --------------------------------------------
# Paras definition (only change these parameters)
# --------------------------------------------

# model information
model_struct_i = models.UnetGenerator(input_nc=3, output_nc=3, num_downs=8)
model_weights_i = 'training_pix2pix_denoiser_denoiser_f32.pth'

# dataset inormation
dataset_dir = 'custom/dataset'
float_model = 'custom/float_model'

# data shape information
channel_i = 3
height_i = 256
width_i = 256


# other parameters
DIVIDER = '-----------------------------------------'


# --------------------------------------------
# Fucntions definition
# --------------------------------------------


class FacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_dir = os.path.join(root_dir, f'{mode}A')  # trainA or testA
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # RGB

        if self.transform:
            image = self.transform(image)

        return image 

def get_dataset(batchsize):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),      
        transforms.ToTensor(),               
    ])

    test_dataset = FacadesDataset(root_dir='./data/facades', transform=transform, mode='test')

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return test_loader