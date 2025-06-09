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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import torch



NOISE_FACTOR = 0.2
# Function to add noise to images
def add_noise(images, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # Ensure values stay within [0, 1]
    return noisy_images

# only used if script is run as 'main' from command line
def main():

    # read images from /images/clean
    clean_dir = 'images/clean'
    # add noise to images and save to /images/denoised
    noisy_dir = 'images/denoised'

    # use cv2 to read images
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
    for filename in os.listdir(clean_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(clean_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            # add noise to image
            noisy_img = add_noise(torch.tensor(img / 255.0).unsqueeze(0).unsqueeze(0))
            noisy_img = noisy_img.squeeze(0).squeeze(0).numpy() * 255.0
            noisy_img = noisy_img.astype(np.uint8)
            noisy_img_path = os.path.join(noisy_dir, filename)
            cv2.imwrite(noisy_img_path, noisy_img)
    print(f'Noisy images saved to {noisy_dir}')

if __name__ == '__main__':
  main()

