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
from common import *
from custom.configs import *
import numpy as np
from torch.utils.data import Subset
import time
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quantize(build_dir, quant_mode, batchsize, calib_size=500, calib_iterations=100):
    """
    模型量化函数 - 修复量化参数检查错误
    :param build_dir: 构建目录
    :param quant_mode: 量化模式 ('calib' 或 'test')
    :param batchsize: 批大小
    :param calib_size: 校准数据集大小 (默认500个样本)
    :param calib_iterations: 校准迭代次数 (默认100次)
    """
    # 设置量化模型路径
    quant_model_dir = os.path.join(build_dir, 'quant_model')
    os.makedirs(quant_model_dir, exist_ok=True)
    logger.info(f"Quantization output directory: {quant_model_dir}")
    
    # 设备检测
    if torch.cuda.is_available():
        logger.info(f'You have {torch.cuda.device_count()} CUDA devices available')
        for i in range(torch.cuda.device_count()):
            logger.info(f' Device {i}: {torch.cuda.get_device_name(i)}')
        logger.info('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        logger.warning('No CUDA devices available..selecting CPU')
        cuda_home = os.environ.get('CUDA_HOME', '')
        if cuda_home:
            logger.warning(f'Using CUDA_HOME: {cuda_home}')
        device = torch.device('cpu')
    
    # 加载模型结构
    model = model_struct_i.to(device)
    logger.info(f"Model structure loaded: {type(model).__name__}")
    
    # 加载模型权重
    float_model_path = os.path.join(float_model, model_weights_i)
    logger.info(f"Loading model weights from: {float_model_path}")
    model.load_state_dict(torch.load(float_model_path, map_location=device))
    model.eval()
    logger.info("Model weights loaded and set to evaluation mode")
    
    # 测试模式下批大小固定为1
    if quant_mode == 'test':
        batchsize = 1
        logger.info(f"Test mode: setting batchsize to 1")
    
    # 创建虚拟输入
    rand_in = torch.randn([batchsize, channel_i, height_i, width_i]).to(device)
    logger.info(f"Created random input tensor: shape={rand_in.shape}")
    
    # 初始化量化器
    logger.info(f"Initializing quantizer in {quant_mode} mode")

    # specify quant config

    # my_quant_config = {
    #     'calib_bit': 8,
    #     'calib_method': 'percentile',  # 使用百分位数方法进行校准
    #     'quantize_weight': True,
    #     'weight_bit': 8,
    #     'quantize_activation': True,
    #     'exponent_range': [-8, 7]  # 添加硬件支持的指数范围约束
    # }
    #my_quant_config_file = os.path.join(build_dir,'quant_config/my_quant.json')
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model_dir)#, quant_config_file = my_quant_config_file)#, quant_config = my_quant_config) 
    quantized_model = quantizer.quant_model
    logger.info("Quantizer initialized")
    
    # 获取完整测试数据集
    test_loader = get_dataset(batchsize)
    logger.info(f"Test dataset loaded with {len(test_loader.dataset)} samples")
    
    # 根据模式选择数据集
    if quant_mode == 'calib':
        # 从测试数据集中创建校准子集
        calib_loader = create_calib_subset(test_loader, calib_size, batchsize)
        logger.info(f"Calibration subset created with {len(calib_loader.dataset)} samples")
        
        # 运行校准过程
        calibrate_model(quantized_model, calib_loader, device, calib_iterations)
        
        # 导出量化配置
        quantizer.export_quant_config()
        logger.info(f"Calibration completed. Quant config saved to {quant_model_dir}")
        
        # 使用正确方法检查量化参数 - 传递目录路径而非模型对象
        check_quantization_params(quant_model_dir)
        
    else:  # test模式
        # 使用原有test函数测试量化模型
        test_loss = test(quantized_model, test_loader)
        logger.info(f'Quantized Test Loss: {test_loss:.4f}')
        
        # 导出部署模型
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model_dir)
        logger.info(f"Quantized model (XModel) saved to {quant_model_dir}")
    
    return quant_model_dir

def create_calib_subset(full_loader, calib_size, batchsize):
    """
    创建校准子集 - 确保数据分布合理
    """
    full_dataset = full_loader.dataset
    logger.info(f"Full dataset size: {len(full_dataset)}")
    
    # 确保选择有代表性的样本
    calib_size = min(calib_size, len(full_dataset))
    indices = np.random.choice(len(full_dataset), calib_size, replace=False)
    logger.info(f"Selected {calib_size} samples for calibration")
    
    # 检查数据分布
    sample_data = [full_dataset[i][0] for i in indices[:10]]
    mean_val = torch.stack(sample_data).mean().item()
    std_val = torch.stack(sample_data).std().item()
    logger.info(f"Calibration data stats: mean={mean_val:.4f}, std={std_val:.4f}")
    
    calib_subset = Subset(full_dataset, indices)
    
    return torch.utils.data.DataLoader(
        calib_subset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=full_loader.num_workers if hasattr(full_loader, 'num_workers') else 0,
        pin_memory=full_loader.pin_memory if hasattr(full_loader, 'pin_memory') else False
    )

def calibrate_model(model, calib_loader, device, iterations=100):
    """
    优化后的校准过程 - 解决量化参数变化过大的问题
    """
    model.eval()
    logger.info(f"Running optimized calibration with {len(calib_loader.dataset)} samples, {iterations} iterations...")
    
    # 添加预热阶段
    logger.info("Starting warm-up phase...")
    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= 5:  # 5个批次预热
                break
            if isinstance(data, (list, tuple)):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            _ = model(inputs)
    
    # 主校准阶段
    logger.info("Starting main calibration phase...")
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= iterations:
                break
                
            if isinstance(data, (list, tuple)):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            
            # 添加小批量处理
            if inputs.size(0) > 1:
                # 分批处理避免内存溢出
                for j in range(0, inputs.size(0), 4):  # 每批4个样本
                    end_idx = min(j+4, inputs.size(0))
                    _ = model(inputs[j:end_idx])
            else:
                _ = model(inputs)
            
            # 添加延迟避免资源争用
            if i % 10 == 0:
                time.sleep(0.1)
            
            # 进度显示
            if (i + 1) % 10 == 0 or (i + 1) == min(iterations, len(calib_loader)):
                elapsed = time.time() - start_time
                logger.info(f"Calibration: {i+1}/{min(iterations, len(calib_loader))} batches "
                      f"({elapsed:.1f}s elapsed)")
    
    logger.info(f"Calibration completed in {time.time()-start_time:.1f} seconds.")

def check_quantization_params(quant_model_dir):
    """
    通过解析生成的量化配置文件检查量化参数
    """
    quant_config_path = os.path.join(quant_model_dir, "quant_info.json")
    
    if not os.path.exists(quant_config_path):
        logger.warning(f"Quantization config file not found at {quant_config_path}")
        return
    
    logger.info(f"\nChecking quantization parameters from config file: {quant_config_path}")
    
    try:
        with open(quant_config_path, 'r') as f:
            quant_config = json.load(f)
        
        all_ok = True
        problematic_layers = []
        
        # 检查所有量化层 - 确保只处理字典类型的条目
        for layer_name, layer_info in quant_config.items():
            # 跳过非字典类型的条目
            if not isinstance(layer_info, dict):
                logger.warning(f"Skipping non-dictionary entry for layer: {layer_name}")
                continue
                
            if 'quant' in layer_info and isinstance(layer_info['quant'], dict):
                quant_info = layer_info['quant']
                
                # 检查scale值
                if 'scale' in quant_info:
                    scale = quant_info['scale']
                    if not isinstance(scale, (int, float)):
                        logger.warning(f"Invalid scale type for layer {layer_name}: {type(scale)}")
                        continue
                        
                    if scale < 1e-6 or scale > 1e6:
                        logger.warning(f"Layer {layer_name} has extreme scale value: {scale:.6f}")
                        all_ok = False
                        problematic_layers.append(layer_name)
                
                # 检查zero_point值
                if 'zero_point' in quant_info:
                    zero_point = quant_info['zero_point']
                    if not isinstance(zero_point, (int, float)):
                        logger.warning(f"Invalid zero_point type for layer {layer_name}: {type(zero_point)}")
                        continue
                        
                    if abs(zero_point) > 128:
                        logger.warning(f"Layer {layer_name} has large zero_point: {zero_point}")
                        all_ok = False
                        problematic_layers.append(layer_name)
            else:
                logger.debug(f"Skipping layer without quant info: {layer_name}")
        
        if all_ok:
            logger.info("All quantization parameters are within expected ranges.")
        else:
            logger.warning(f"Found {len(problematic_layers)} layers with problematic quantization parameters.")
            logger.warning("This may cause deployment issues. Consider:")
            logger.warning("1. Increasing calibration data size")
            logger.warning("2. Adding more calibration iterations")
            logger.warning("3. Using a different calibration method")
            logger.warning("4. Adding batch normalization layers")
            logger.warning("5. Using quantization-aware training")
            
            # 保存问题层信息
            problem_file = os.path.join(quant_model_dir, "problematic_layers.txt")
            with open(problem_file, 'w') as f:
                f.write("\n".join(problematic_layers))
            logger.info(f"Problematic layers saved to: {problem_file}")
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {quant_config_path}: {str(e)}")
        logger.error("The quant_info.json file may be corrupted or incomplete.")
    except Exception as e:
        logger.error(f"Error checking quantization parameters: {str(e)}", exc_info=True)


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