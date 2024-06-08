#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,4

python -u train.py --config configs/densenet.yaml