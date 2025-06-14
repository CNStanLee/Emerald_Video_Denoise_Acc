conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

python -u train.py

python -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


source compile.sh u50 ${BUILD} ${LOG} --debug


python -u target.py --target u50 --num_images 10    -d ${BUILD} 2>&1 | tee ${LOG}/target_u50.log

