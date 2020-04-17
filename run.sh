#!/bin/bash

#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_

module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list

VIRTENV=RES_FLOWS
VIRTENV_ROOT=~/virtualenvs



#pip install torch
#pip install torchvision
#pip install torchsummary
#pip install tqdm
#pip install scikit-image


python train_img.py \
 --data custom \
 --dataset 17 \
 --train_centers 1 \
 --val_centers 1 \
 --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --val_split 0.2 \
 --imagesize 256 \
 --batchsize 8 \
 --val-batchsize 8 \
 --actnorm True \
 --act elu \
 --wd 0 \
 --update-freq 5 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first True \
 --save experiments/examode256 \
 --nblocks 16-16-16
 #--nblocks 16-16-16-16-16-16 \
 #--debug





# python train_img.py \
# --data mnist \
# --imagesize 28 \
# --actnorm True \
# --wd 0 \
# --save experiments/mnist


