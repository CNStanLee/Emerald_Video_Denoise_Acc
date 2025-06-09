# Vitis-AI UNet

## Purpose
This project is to rebuild Emmet Murphy's workflow aka transfering a UNet to Alveo U50 for denosing

## Preparation
Emmet's project was built up using Vitis-AI 1.4.1,
we dont know if things build with this version will happy with the things built with VAI 3.0/3.5, thus we use VAI 1.4.1 to rebuild the flow.
so we need to config things below:
### Vitis 2021.1
VAI 1.4.1 only work with Vitis 2021.1
### VAI 1.4.1
clone from VAI github and switch to 1.4.1
### Docker Image
cd /home/changhong/prj/vai_unet/0_follow_tutorial/09-mnist_pyt/files
./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978
We'd better to mannuly choose vai version to avoid any version problems

## How to run

```shell
Vitis-AI /workspace > conda activate vitis-ai-pytorch
(vitis-ai-pytorch) Vitis-AI /workspace >
```

```shell
(vitis-ai-pytorch) Vitis-AI /workspace > source run_all.sh
```


## How to deploy on Alveo server

## Alveo U50

**Note:** The U50 should be flashed with the correct deployment shell, and this should have been done in the 'Preparing the host machine and target boards' section above.

The following steps should be run from inside the Vitis-AI Docker container:

  + Ensure that Vitis-AI's PyTorch conda environment is enabled (if not, the run `conda activate vitis-ai-pytorch`).

  + Run `source setup.sh DPUCAHX8H` which sets environment variables to point to the correct overlay for the U50. The complete steps to run are as follows:


```shell
conda activate vitis-ai-pytorch
source setup.sh DPUCAHX8H
cd build/target_u50
/usr/bin/python3 app_mt.py -m CNN_u50.xmodel
```

The console output will be like this:

```shell
(vitis-ai-pytorch) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py -m CNN_u50.xmodel
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  CNN_u50.xmodel
-------------------------------
Pre-processing 10000 images...
-------------------------------
Starting 1 threads...
-------------------------------
Throughput=14362.58 fps, total frames = 10000, time=0.6963 seconds
Correct:9877, Wrong:123, Accuracy:0.9877
-------------------------------
```

Perfromance can be slightly increased by increasing the number fo threads:

```shell
(vitis-ai-pytorch) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py -m CNN_u50.xmodel -t 6
Command line options:
 --image_dir :  images
 --threads   :  6
 --model     :  CNN_u50.xmodel
-------------------------------
Pre-processing 10000 images...
-------------------------------
Starting 6 threads...
-------------------------------
Throughput=16602.34 fps, total frames = 10000, time=0.6023 seconds
Correct:9877, Wrong:123, Accuracy:0.9877
-------------------------------
```