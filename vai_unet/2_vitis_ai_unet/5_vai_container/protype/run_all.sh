conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# # run training
# python -u train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log


# # quantize & export quantized model
python -u quantize_unet.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u quantize_unet.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# compile for target boards
# source compile.sh zcu102 ${BUILD} ${LOG}
# source compile.sh zcu104 ${BUILD} ${LOG}
source compile.sh u50 ${BUILD} ${LOG}
# source compile.sh vck190 ${BUILD} ${LOG}

# make target folders
# python -u target.py --target zcu102 -d ${BUILD} 2>&1 | tee ${LOG}/target_zcu102.log
# python -u target.py --target zcu104 -d ${BUILD} 2>&1 | tee ${LOG}/target_zcu104.log
# python -u target.py --target vck190 -d ${BUILD} 2>&1 | tee ${LOG}/target_vck190.log
python -u target.py --target u50    -d ${BUILD} 2>&1 | tee ${LOG}/target_u50.log

