#!/bin/bash
source activate torch

# Availible GPU
device=`/home/xma/tools/bin/free-gpu`
echo "GPU:${device}"

# Necessary environment variables
LD_LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64:/opt/NVIDIA/cuda-8/lib64/
CPATH=/export/b18/xma/libs/cudnn-6/cuda/include
LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64
code_path="/export/b03/cszu/MT-HW/hw4"

#mkdir -p model
CUDA_VISIBLE_DEVICES=${device} python ${code_path}/train.py \
  --optimizer Adam -lr 1e-2 --batch_size 48 --estop 1e-3\
  --gpuid 0 \
  --data_file ${code_path}/data/hw4_data.bin \
  --model_file model

