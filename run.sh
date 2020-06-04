#!/bin/bash

##SBATCH -N 1
##SBATCH -t 8:00:00
##SBATCH -p gpu_titanrtx

module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list

#VIRTENV=RES_FLOWS
#VIRTENV_ROOT=~/virtualenvs


#rm -rf experiments/*
#pip install torch
#pip install torchvision
#pip install torchsummary
#pip install tqdm
#pip install scikit-image


# python train_img.py \
#  --data custom \
#  --dataset 17 \
#  --train_centers 1 \
#  --val_centers 2 \
#  --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
#  --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
#  --val_split 0.2 \
#  --imagesize 256 \
#  --batchsize 4 \
#  --val-batchsize 1 \
#  --actnorm True \
#  --nbits 8 \
#  --act swish \
#  --update-freq 1 \
#  --n-exact-terms 8 \
#  --fc-end False \
#  --squeeze-first False \
#  --factor-out True \
#  --save experiments/gmm11 \
#  --nblocks 16 \
#  --gmmsteps 1000 \
#  --vis-freq 50 \
#  --nepochs 5
#  #--resume /home/rubenh/examode/color-information/experiments/gmm12/models/most_recent.pth \

 

# python train_img.py \
#  --data custom \
#  --dataset 17 \
#  --train_centers 2 \
#  --val_centers 3 \
#  --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
#  --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
#  --val_split 0.2 \
#  --imagesize 256 \
#  --batchsize 4 \
#  --val-batchsize 8 \
#  --actnorm True \
#  --nbits 8 \
#  --act swish \
#  --update-freq 1 \
#  --n-exact-terms 8 \
#  --fc-end False \
#  --squeeze-first False \
#  --factor-out True \
#  --save experiments/gmm12 \
#  --nblocks 16 \
#  --gmmsteps 1000 \
#  --vis-freq 50 \
#  --nepochs 5
 
 python train_img.py \
 --data custom \
 --dataset 17 \
 --train_centers 3 \
 --val_centers 4 \
 --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --val_split 0.2 \
 --imagesize 256 \
 --batchsize 4 \
 --val-batchsize 8 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --save experiments/gmm13 \
 --nblocks 16 \
 --gmmsteps 1000 \
 --vis-freq 50 \
 --nepochs 5
 
  python train_img.py \
 --data custom \
 --dataset 17 \
 --train_centers 4 \
 --val_centers 4 \
 --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --val_split 0.2 \
 --imagesize 256 \
 --batchsize 4 \
 --val-batchsize 8 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --save experiments/gmm14 \
 --nblocks 16 \
 --gmmsteps 1000 \
 --vis-freq 50 \
 --nepochs 5


 #--nblocks 16-16-16-16-16-16 \
 #--debug