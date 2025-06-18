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
# custom imports
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
# from common import *
# import custom.models as models

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models_cle_mod_vai import *

# --------------------------------------------
# Paras definition (only change these parameters)
# --------------------------------------------

# model information
model_struct_i = UnetGenerator(input_nc=3, output_nc=3, num_downs=8)
model_weights_i = 'cle_mod_vai_unet_facades_f32.pth'

# dataset inormation
dataset_dir = 'facades'
float_model = 'build/float_model'

# data shape information
channel_i = 3
height_i = 256
width_i = 256


# other parameters
DIVIDER = '-----------------------------------------'

criterion = nn.MSELoss()
# dataset_path = 'facades'



BATCH_SIZE = 64
EPOCHS = 20
NOISE_FACTOR = 0.2
LEARNING_RATE = 0.001

def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Ensure values stay within [0, 1]
    return noisy_images
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

    test_dataset = FacadesDataset(root_dir='facades', transform=transform, mode='test')

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return test_loader

def test(model, testloader, device):
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for images in testloader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            test_loss += loss.item()
    return test_loss / len(testloader)


def quantize(build_dir,quant_mode,batchsize):
    # set build path
    quant_model = build_dir + '/quant_model'
    # check device
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')
    # load trained model
    model = model_struct_i.to(device)  # change to
    # load model weights
    model.load_state_dict(torch.load(os.path.join(float_model,model_weights_i)))
    
    # override batchsize if in test mode
    if (quant_mode=='test'):
        batchsize = 1

    # make dummy input
    rand_in = torch.randn([batchsize, channel_i, height_i, width_i])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model
    
    # get test loader
    test_loader = get_dataset(batchsize)
    test_loss = test(quantized_model, test_loader, device) # add device
    print(f'Test Loss: {test_loss:.4f}')

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return

# --------------------------------------------
# Main
# --------------------------------------------
def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return



if __name__ == '__main__':
    run_main()