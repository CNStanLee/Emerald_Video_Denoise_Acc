# Acc protype v3
## What this will do
This is an acc protype which will denoise Cifar-10 with random noise using Alveo U50.
We can use this protype to develope API, when the final model ready I will replace the acc while keeping same interface.
## Req
- Vitis-AI 1.4.1: could git from official github, the original docker image is expired, we mannully target to: xilinx/viti-ai-cpu:1.4.1.978
## How to run this
(bash): basic enviroment
(vai): vitis-ai docker
- Firstly start Vitis-AI env (I have already created one on the server in home/Changhong_workspace/Vitis-AI), cd into this folder and then run the docker
```shell
(bash) cd /home/emurphy/changhong_workspace/Vitis-AI
(bash) source start_vai_141.sh
```
- Put this deployable folder (release_deployable) into the viti-ai workspace (same path of docker_run.sh), there is already one inside it
- In the vitis-ai docker, activate the DPU, you must activate it everytime you start the docker 
```shell
(vai) conda activate vitis-ai-pytorch
(vai) source /workspace/setup/alveo/setup.sh DPUCAHX8H
```
- Run the application
```shell
(vai) cd /workspace/release_deployable/protype3/
(vai) python app_mt.py
```

- Check result
The result of througput, latency, PSNR, MSE will be shown in the bash.
The denoised result is in /workspace/release_deployable/protype3/images/denoised
