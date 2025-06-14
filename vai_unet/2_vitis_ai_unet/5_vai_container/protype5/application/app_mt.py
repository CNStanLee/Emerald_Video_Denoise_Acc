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

_divider = '-------------------------------'

image_width = 256
image_height = 256
image_channels = 3

image_path = 'images/'
noisy_image_path = image_path + 'noisy/'
clean_image_path = image_path + 'clean/'
denoised_image_path = image_path + 'denoised/'


def preprocess_fn(image_path, fix_scale):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Use COLOR instead of GRAYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # OpenCV uses BGR by default
    
    # Resize to 
    if image.shape != (image_height, image_width, image_channels):
        image = cv2.resize(image, (image_height, image_width))
    
    # Normalize to [0, 1] and apply scaling
    image = image * (1.0 / 255.0) * fix_scale
    
    # Convert to int8 (quantization)
    image = image.astype(np.int8)
    
    return image



def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, img):
    '''Run DPU inference for CIFAR-10 (32x32x3 RGB input/output)'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)   
    output_ndim = tuple(outputTensors[0].dims) 
    print(f"Input shape: {input_ndim}, Output shape: {output_ndim}")

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids = []
    ids_max = 5
    outputData = []
    
    # Initialize output buffers (for CIFAR-10's 32x32x3 output)
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    
    # Initialize a list to store output images
    output_images = []

    while count < n_of_images:
        runSize = min(batchSize, n_of_images - count)

        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        for j in range(runSize):
            inputData[0][j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        # Run DPU async
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count += runSize

        # Process completed jobs if queue is full or all images are processed
        if count < n_of_images and len(ids) < ids_max - 1:
            continue

        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]

            for j in range(ids[index][1]):
                output_img = outputData[index][0][j] 
                output_images.append(output_img.copy())  # Save to output list
                out_q[write_index] = output_img  # Original logic (if needed)
                write_index += 1

        ids = []

    return output_images  




def app(image_dir,threads,model):

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    ''' preprocess images '''
    print (_divider)
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    print (_divider)
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print (_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    # save denoised images
    clean_dir = 'images/clean'
    noisy_dir = 'images/noisy'
    denoised_dir = 'images/denoised'
    if not os.path.exists(denoised_dir):
        os.makedirs(denoised_dir)

    # Get output scale (if DPU output is quantized)
    output_fixpos = all_dpu_runners[0].get_output_tensors()[0].get_attr("fix_point")
    output_scale = 2 ** output_fixpos if output_fixpos is not None else 1.0

    average_psnr = 0
    average_mse = 0

    for i in range(len(out_q)):
        # 1. De-quantize and scale DPU output to [0, 255]
        denoised_img = (out_q[i] / output_scale).clip(0, 1) * 255
        denoised_img = denoised_img.astype(np.uint8)  # Convert to uint8 for image saving
        
        # 2. Save denoised image (as RGB)
        output_path = os.path.join(denoised_dir, f'denoised_{i}.png')
        cv2.imwrite(output_path, cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR
        
        # 3. Load original clean image (as RGB)
        clean_image_path = os.path.join(clean_dir, f'clean_{i}.png')
        clean_image = cv2.imread(clean_image_path, cv2.IMREAD_COLOR)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # load noisy image (as RGB)
        noisy_image_path = os.path.join(noisy_dir, f'noisy_{i}.png')
        noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_COLOR)
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # calculate noisy image MSE and PSNR
        noisy_mse = np.mean((clean_image - noisy_image) ** 2)
        noisy_psnr = 20 * np.log10(255.0 / np.sqrt(noisy_mse)) if noisy_mse != 0 else float('inf')

        # 4. Calculate MSE and PSNR (RGB channels)
        mse = np.mean((clean_image - denoised_img) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
        
        average_psnr += psnr
        average_mse += mse

        average_noisy_psnr += noisy_psnr
        average_noisy_mse += noisy_mse

    average_psnr /= len(out_q)
    average_mse /= len(out_q)
    average_noisy_psnr /= len(out_q)
    average_noisy_mse /= len(out_q)
    average_psnr_improvement =  average_psnr - average_noisy_psnr
    average_mse_improvement = average_mse - average_noisy_mse

    print('Denoised images saved to:', denoised_dir)
    print('Average Denoised PSNR (RGB): %.2f dB' % average_psnr)
    print('Average Denoised MSE (RGB): %.4f' % average_mse)
    print('Average Noisy PSNR (RGB): %.2f dB' % average_noisy_psnr)
    print('Average Noisy MSE (RGB): %.4f' % average_noisy_mse)
    print('Average PSNR Improvement: %.2f dB' % average_psnr_improvement)
    print('Average MSE Improvement: %.4f' % average_mse_improvement)
    print('Done.')
    print(_divider)




# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images/noisy', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='UnetGenerator_u50.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()

