#!/bin/bash

source activate torch

python decode_cloze.py --data_file data/hw4_data.bin --model_file model.nll_5.11.epoch_4 --output_file output_test --test_file data/test.en.txt.cloze
