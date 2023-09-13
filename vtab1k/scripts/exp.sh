#!/bin/sh
devices=$1
method=$2
dim=$3
scale=$4
DATASET=$5
lr=$6
wd=$7
dropout=$8
bsize=$9
seed=42

CUDA_VISIBLE_DEVICES=$devices  python train.py --dataset $DATASET --method $method --dim $dim --scale $scale --lr $lr --seed $seed --wd $wd --dropout $dropout --bsize $bsize