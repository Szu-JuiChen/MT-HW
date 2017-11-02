#!/bin/bash
set -e

# Activate virtual enviroment
source activate NeuralHawkes

# Availible GPU
device=`/home/xma/tools/bin/free-gpu`
echo "GPU:${device}"

# Necessary environment variables
LD_LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64:/opt/NVIDIA/cuda-8/lib64/
CPATH=/export/b18/xma/libs/cudnn-6/cuda/include
LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64

neural_hawkes='/export/b18/xma/neural_hawkes/NeuralHawkesRumuorStance'
data_path="${neural_hawkes}/data/plk/sydney.plk"
log_path="`pwd`/log.1"
echo $log_path
mkdir -p $log_path

# run 
CUDA_VISIBLE_DEVICES=${device} python ${neural_hawkes}/train.py  \
  --ModelType NeuralHawkes \
  --DataSet ${data_path} \
  --Seed 12345 \
  --MultiSample 10 \
  --BatchSize 1 \
  --DimModel 32 \
  --MaxEpoch 20 \
  --CoefToken 1.0\
  --CoefGen 0.1 0.1 0.01\
  --CoefDis 1 \
  --SaveLog ${log_path} \
  --GPU \
  --Test


