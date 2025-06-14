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


def generate_images(dset_dir, num_images, dest_dir):

    print('Generating', num_images, 'noisy images in', dest_dir)

    # copy the folder and files in build/datapack to dest_dir
    src_dir = 'build/facades_test'
    
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

    # copy readme file
    shutil.copy(os.path.join(app_dir, 'readme.md'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/UnetGenerator_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)
    
    # copy checksum file
    checksum_path = comp_dir + '/md5sum.txt'
    print('Copying checksum file from',checksum_path,'...')
    shutil.copy(checksum_path, target_dir)

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

