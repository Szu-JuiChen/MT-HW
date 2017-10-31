#!/bin/bash

source activate NeuralHawkes

python decode_cloze.py --data_file data/hw4_data.bin --model_file experiment/model.nll_5.18.epoch_6 --output_file output_test --test_file data/test.en.txt.cloze
