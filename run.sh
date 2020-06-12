#!/bin/bash

##SBATCH -N 1
##SBATCH -t 8:00:00
##SBATCH -p gpu_titanrtx

module purge
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list

# Setting ENV variables
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export HOROVOD_WITH_PYTORCH=1
#VIRTENV=RES_FLOWS
#VIRTENV_ROOT=~/virtualenvs


#rm -rf experiments/*
#pip install horovod
#pip install torch
#pip install torchvision
#pip install torchsummary
#pip install tqdm
#pip install scikit-image
 
 
 #mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python -u train_img.py \
 #mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train_img.py \
 python train_img.py \
 --data custom \
 --dataset 17 \
 --train_centers 1 \
 --val_centers 4 \
 --train_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --valid_path /nfs/managed_datasets/CAMELYON17/training/center_XX \
 --val_split 0.2 \
 --imagesize 256 \
 --batchsize 32 \
 --val-batchsize 32 \
 --actnorm True \
 --nbits 8 \
 --act swish \
 --update-freq 1 \
 --n-exact-terms 8 \
 --fc-end False \
 --squeeze-first False \
 --factor-out True \
 --save experiments/test \
 --nblocks 16 \
 --vis-freq 50 \
 --resume /home/rubenh/examode/color-information/experiments/gmm1/models/most_recent.pth \
 --nepochs 1 




 #--nblocks 16-16-16-16-16-16 \
 #--debug