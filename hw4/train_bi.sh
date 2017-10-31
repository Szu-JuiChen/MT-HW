#!/bin/bash
source activate NeuralHawkes
mkdir -p model
python train_bi.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model

