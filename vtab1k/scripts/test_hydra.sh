#!/bin/sh
echo "start"
devices=0
method='hydra_both'
dim=2
# dataset=(cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)
dataset=(cifar)
lr=-1
dropout=-1.0
bsize=-1
scale=0
wd=-1

for data in ${dataset[@]};
do  
    logpath=./logs/test/$method
    if ! [ -d "$logpath" ]; then
        mkdir -p $logpath
    fi
    CUDA_VISIBLE_DEVICES=$devices python test.py --method $method --dim $dim --scale $scale  --dataset $data --lr $lr --wd $wd --dropout $dropout --bsize $bsize 2>&1 | tee $logpath/$data.log
done