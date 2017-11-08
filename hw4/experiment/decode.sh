#!/bin/bash

source activate NeuralHawkes

device=`/home/xma/tools/bin/free-gpu`
echo "GPU:${device}"

# Necessary environment variables
LD_LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64:/opt/NVIDIA/cuda-8/lib64/
CPATH=/export/b18/xma/libs/cudnn-6/cuda/include
LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64
code_path="/export/b18/xma/course/machine_translation/MT-HW/hw4"

#mkdir -p model
CUDA_VISIBLE_DEVICES=${device} python ../decode_cloze.py \
  --data_file ../data/hw4_data.bin \
  --model_file $@\
  --output_file ../output_pun \
  --test_file ../data/test.en.txt.cloze
