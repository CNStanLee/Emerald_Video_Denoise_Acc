# export FINN_XILINX_PATH=/tools/Xilinx
# export FINN_XILINX_VERSION=2022.2
# source /tools/Xilinx/Vivado/2022.2/settings64.sh
# export VIVADO_PATH=/tools/Xilinx/Vivado/2022.2
# export JUPYTER_PORT=8886
# export NVIDIA_VISIBLE_DEVICES=all


cd /home/changhong/prj/vai_unet/0_follow_tutorial/09-mnist_pyt/files
# ./docker_run.sh xilinx/vitis-ai-gpu:latest
# ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
# docker pull xilinx/vitis-ai-cpu:1.4.1.978
./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978