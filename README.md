# Residual Flows for Invertible Generative Modeling 

- Flow-based generative models parameterize probability distributions through an
invertible transformation and can be trained by maximum likelihood. Invertible
residual networks provide a flexible family of transformations where only Lipschitz
conditions rather than strict architectural constraints are needed for enforcing
invertibility.
- This application concerns the invertible mapping of the color information in histopathology Whole Slide Image patches

## Requirements

```
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
pip install requirements.txt
```


## Preprocessing

CAMELYON17 256 x 256:
```
See Examode preprocessing
```

ImageNet:
1. Follow instructions in `preprocessing/create_imagenet_benchmark_datasets`.
2. Convert .npy files to .pth using `preprocessing/convert_to_pth`.
3. Place in `data/imagenet32` and `data/imagenet64`.

CelebAHQ 64x64 5bit:

1. Download from https://github.com/aravindsrinivas/flowpp/tree/master/flows_celeba.
2. Convert .npy files to .pth using `preprocessing/convert_to_pth`.
3. Place in `data/celebahq64_5bit`.

CelebAHQ 256x256:
```
# Download Glow's preprocessed dataset.
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -C data/celebahq -xvf celeb-tfr.tar
python extract_celeba_from_tfrecords
```



## Density Estimation Experiments

CAMELYON17 256 x 256

```
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
 ```

MNIST:
```
python train_img.py --data mnist --imagesize 28 --actnorm True --wd 0 --save experiments/mnist
```

CIFAR10:
```
python train_img.py --data cifar10 --actnorm True --save experiments/cifar10
```

ImageNet 32x32:
```
python train_img.py --data imagenet32 --actnorm True --nblocks 32-32-32 --save experiments/imagenet32
```

ImageNet 64x64:
```
python train_img.py --data imagenet64 --imagesize 64 --actnorm True --nblocks 32-32-32 --factor-out True --squeeze-first True --save experiments/imagenet64
```

CelebAHQ 256x256:
```
python train_img.py --data celebahq --imagesize 256 --nbits 5 --actnorm True --act elu --batchsize 8 --update-freq 5 --n-exact-terms 8 --fc-end False --factor-out True --squeeze-first True --nblocks 16-16-16-16-16-16 --save experiments/celebahq256
```

## Pretrained Models

Model checkpoints can be downloaded from [releases](https://github.com/rtqichen/residual-flows/releases/latest).

Use the argument `--resume [checkpt.pth]` to evaluate or sample from the model. 

Each checkpoint contains two sets of parameters, one from training and one containing the exponential moving average (EMA) accumulated over the course of training. Scripts will automatically use the EMA parameters for evaluation and sampling.

## Based on paper
```
@inproceedings{chen2019residualflows,
  title={Residual Flows for Invertible Generative Modeling},
  author={Chen, Ricky T. Q. and Behrmann, Jens and Duvenaud, David and Jacobsen, J{\"{o}}rn{-}Henrik},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2019}
}
```
