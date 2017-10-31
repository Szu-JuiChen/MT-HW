#!/bin/bash

source activate NeuralHawkes

python decode_cloze.py --data_file data/hw4_data.bin --model_file model.nll_-0.24.epoch_4 --output_file output_test --test_file data/test.en.txt.cloze
