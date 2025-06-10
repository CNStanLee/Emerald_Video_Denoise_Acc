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
Make the target folder
Copies images, application code and compiled xmodel to 'target'
'''

'''
Author: Mark Harvey
'''

import torch
import torchvision

import argparse
import os
import shutil
import sys
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from common import gen_transform
import matplotlib.pyplot as plt

import os
import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)



DIVIDER = '-----------------------------------------'

NOISE_FACTOR = 0.2
# Function to add noise to images
def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Ensure values stay within [0, 1]
    return noisy_images




def generate_images(dset_dir, num_images, dest_dir):

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR10 是 RGB 图像
    # ])

    
    # testset = datasets.CIFAR10(root='0_data/', download=True, train=False, transform=transform)

    
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)


    # print('Generating', num_images, 'noisy images in', dest_dir)
    # dataiter = iter(test_loader)
    # # 创建目录
    # os.makedirs(dest_dir + '/clean', exist_ok=True)
    # os.makedirs(dest_dir + '/noisy', exist_ok=True)
    # clean_dir = dest_dir + '/clean'
    # noisy_dir = dest_dir + '/noisy'
      
    # for i in tqdm(range(num_images)):
    #     # 获取图像数据
    #     image, label = next(dataiter)
    #     image = image[0]  # 形状: (3, 32, 32)
    #     noisy_image = add_noise(image)  # 形状: (3, 32, 32)

    #     # 转换为 numpy 并调整通道顺序为 (H, W, C)
    #     img = image.permute(1, 2, 0).numpy()  # 形状: (32, 32, 3)
    #     noisy_img = noisy_image.permute(1, 2, 0).numpy()  # 形状: (32, 32, 3)

    #     # 反标准化：从 [-1, 1] 映射回 [0, 1]
    #     img = (img * 0.5) + 0.5  # [0, 1]
    #     noisy_img = (noisy_img * 0.5) + 0.5  # [0, 1]

    #     # 使用 matplotlib 保存图像（确保与 imshow 一致）
    #     plt.imsave(os.path.join(dest_dir, 'clean', f'clean_{i}.png'), img)
    #     plt.imsave(os.path.join(dest_dir, 'noisy', f'noisy_{i}.png'), noisy_img)
    print('Generating', num_images, 'noisy images in', dest_dir)

    # copy the folder and files in build/datapack to dest_dir
    src_dir = 'build/datapack'
    

    copytree(src_dir, dest_dir)
    
    return


def make_target(build_dir,target,num_images,app_dir):

    dset_dir = build_dir + '/dataset'
    comp_dir = build_dir + '/compiled_model'
    target_dir = build_dir + '/target_' + target

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/Denoise_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)

    # create images
    dest_dir = target_dir + '/images'
    shutil.rmtree(dest_dir, ignore_errors=True)  
    os.makedirs(dest_dir)

    generate_images(dset_dir, num_images, dest_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',  type=str,  default='build', help='Path to build folder. Default is build')
    ap.add_argument('-t', '--target',     type=str,  default='zcu102', choices=['zcu102','zcu104','u50','vck190'], help='Target board type (zcu102,zcu104,u50,vck190). Default is zcu102')
    ap.add_argument('-n', '--num_images', type=int,  default=10000, help='Number of test images. Default is 10000')
    ap.add_argument('-a', '--app_dir',    type=str,  default='application', help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --target       : ', args.target)
    print (' --num_images   : ', args.num_images)
    print (' --app_dir      : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir, args.target, args.num_images, args.app_dir)


if __name__ ==  "__main__":
    main()

