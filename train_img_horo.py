import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc
import sys
import pdb
from glob import glob
from sklearn.utils import shuffle
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image,ImageStat
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
import pyvips
import random
import torch.utils.data.distributed
import horovod.torch as hvd
import cv2
import torch.multiprocessing as mp
import pprint
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets
from torchsummary import summary

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
from lib.GMM import GMM_model as gmm
import lib.image_transforms as imgtf
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts



"""
- Implement in training
- Deploy

"""

# Arguments
parser = argparse.ArgumentParser(description='Residual Flow Model Color Information', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data', type=str, default='custom', choices=[
        'custom'
    ]
)
# mnist
parser.add_argument('--dataroot', type=str, default='data')
## GMM ##
parser.add_argument('--nclusters', type=int, default=4,help='The amount of tissue classes trained upon')

parser.add_argument('--dataset', type=str, default="0", help='Which dataset to use. "16" for CAMELYON16 or "17" for CAMELYON17')
parser.add_argument('--slide_path', type=str, help='Folder of where the training data whole slide images are located', default=None)
parser.add_argument('--mask_path', type=str, help='Folder of where the training data whole slide images masks are located', default=None)
parser.add_argument('--valid_slide_path', type=str, help='Folder of where the validation data whole slide images are located', default=None)
parser.add_argument('--valid_mask_path', type=str, help='Folder of where the validation data whole slide images masks are located', default=None)
parser.add_argument('--slide_format', type=str, help='In which format the whole slide images are saved.', default='tif')
parser.add_argument('--mask_format', type=str, help='In which format the masks are saved.', default='tif')
parser.add_argument('--bb_downsample', type=int, help='Level to use for the bounding box construction as downsampling level of whole slide image', default=7)
parser.add_argument('--log_image_path', type=str, help='Path of savepath of downsampled image with processed rectangles on it.', default='.')
parser.add_argument('--epoch_steps', type=int, help='The hard - coded amount of iterations in one epoch.', default=1000)

# Not used now
#parser.add_argument('--batch_tumor_ratio', type=float, help='The ratio of the batch that contains tumor', default=1)
    
parser.add_argument('--val_split', type=float, default=0.15)
parser.add_argument('--debug', action='store_true', help='If running in debug mode')
parser.add_argument('--fp16_allreduce', action='store_true', help='If all reduce in fp16')
##
parser.add_argument('--imagesize', type=int, default=32)
# 28
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False,help='Learn Lipschitz norms, see paper')

parser.add_argument('--n-power-series', type=int, default=None, help='Amount of power series evaluated, see paper')
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False,help='Factorize dimensions, see paper')
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2,help='Exact terms computed in series estimation, see paper')
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True,help='Neumann gradients, see paper')
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True,help='Memory efficient backprop, see paper')

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=128)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=False)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-idim', type=int, default=8)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=False)
parser.add_argument('--cdim', type=int, default=128)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
# 0
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, help='Use exponential moving averages of parameters at validation.', choices=[True, False], default=False)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid','gmm'], default='gmm')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save_conv', type=eval,help='Save converted images.', default=False)
parser.add_argument('--begin-epoch', type=int, default=0)


parser.add_argument('--nworkers', type=int, default=8)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=1)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=5)
parser.add_argument('--save-every', help='VSave model every so epochs', type=int, default=1)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)



# Assert for now
assert args.batchsize == args.val_batchsize, "Training and Validation batch size must match"

    
# Horovod: initialize library.
hvd.init()
print(f"hvd.size {hvd.size()} hvd.rank {hvd.rank()} hvd.local_rank {hvd.local_rank()}")

def rank00():
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        return True

if rank00():
    # logger
    utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

if rank00():
    logger.info(args)
    
if device.type == 'cuda':
    if rank00():
        logger.info(f'Found {hvd.size()} CUDA devices.')
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    
    if rank00():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}


     
    
def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rescale(tensor):
    """
    Parameters
    ----------
    tensor : Pytorch tensor
        Tensor to be rescaled to [0,1] interval.

    Returns
    -------
    Rescaled tensor.

    """
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor

def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x
    


def open_img(path):
    return np.asarray(Image.open(path))[:, :, 0] / 255


def get_valid_idx(mask_list):
    """ Get the valid indices of masks by opening images in parallel """
    num_cores = multiprocessing.cpu_count()
    data = Parallel(n_jobs=num_cores)(delayed(open_img)(i) for i in mask_list)
    return data

if rank00():
    logger.info('Loading dataset {}'.format(args.data))



class make_dataset(torch.utils.data.Dataset):
    """Make Pytorch dataset."""
    def __init__(self, args,train=True):
        """
        Args:

        """

        self.train = train
        
        if args.mask_path:
            self.train_paths = shuffle(list(zip(sorted(glob(os.path.join(args.slide_path,f'*.{args.slide_format}'))),
                                                sorted(glob(os.path.join(args.mask_path,f'*.{args.mask_format}'))))))
        else:
            self.train_paths = shuffle(sorted(glob(os.path.join(args.slide_path,f'*.{args.slide_format}'))))
        
        print(f"Found {len(self.train_paths)} images")
        if args.valid_slide_path:
            if self.args.mask_path:
                self.valid_paths = shuffle(list(zip(sorted(glob(os.path.join(args.valid_slide_path,f'*.{args.slide_format}'))),
                                                    sorted(glob(os.path.join(args.valid_mask_path,f'*.{args.mask_format}'))))))
            else:
                self.valid_paths = shuffle(sorted(glob(os.path.join(args.valid_slide_path,f'*.{args.slide_format}'))))
        else:
            val_split = int(len(self.train_paths) * args.val_split)
            self.valid_paths = self.train_paths[val_split:]
            self.train_paths = self.train_paths[:val_split]
        
        
        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []
        self.level_used = args.bb_downsample
        self.mag_factor = pow(2, self.level_used)
        self.patch_size = args.imagesize
        # self.tumor_ratio = args.batch_tumor_ratio
        self.log_image_path = args.log_image_path
        self.slide_format = args.slide_format
        
        
    @staticmethod
    def _transform(image,train=True):
        if train:
            return transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                reduce_bits,
                                lambda x: add_noise(x, nvals=2**args.nbits),
                            ])(image)
        else:
            return transforms.Compose([
                                transforms.ToTensor(),
                                reduce_bits,
                                lambda x: add_noise(x, nvals=2**args.nbits),
                            ])(image)
        
        
    def __len__(self):
        return len(self.train_paths)

    def get_bb(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        contours, _ = cv2.findContours(np.array(image_open), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _offset=0
        for i, contour in enumerate(contours):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if contour.shape[0] < 10:
                print(f"Deleted too small contour from {self.cur_wsi_path}")
                del contours[i]
                _offset+=1
                i=i-_offset
        # contours_rgb_image_array = np.array(self.rgb_image)
        # line_color = (255, 150, 150)  
        # cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 1)
        # Image.fromarray(contours_rgb_image_array[...,:3]).save('test.png')

        # self.rgb_image_pil.close()
        # self.wsi.close()
        # self.mask.close()
        
        return contours
    
    def __getitem__(self, train=True):
        if train:
            while not self.contours_train or not self.contours_tumor:
    
                self.cur_wsi_path = random.choice(self.train_paths)
                print(f"Opening {self.cur_wsi_path}...")
                
                if args.mask_path:
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                    self.mask = OpenSlide(self.cur_wsi_path[1])
                else:
                    self.cur_wsi_path = [self.cur_wsi_path]
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])

                
                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
    
                if args.mask_path:
                    self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.mask_image = np.array(self.mask_pil)
                    
                self.contours_train = self.get_bb()
                self.contours = self.contours_train
                
                if args.mask_path:
                    # Get bounding boxes of tumor
                    contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    self.contours_tumor = contours
                else:
                    self.contours_tumor=1   
                    
        else:  
            while not self.contours_valid or not self.contours_tumor:
    
                self.cur_wsi_path = random.choice(self.valid_paths)
                print(f"Opening {self.cur_wsi_path}...")
                
                if args.mask_path:
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                    self.mask = OpenSlide(self.cur_wsi_path[1])
                else:
                    self.cur_wsi_path = [self.cur_wsi_path]
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                
                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
    
                if args.mask_path:
                    self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.mask_image = np.array(self.mask_pil)
                    
                self.contours_valid = self.get_bb()
                self.contours = self.contours_valid
                
                if args.mask_path:
                # Get bounding boxes of tumor
                    contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    self.contours_tumor = contours
                else:
                    self.contours_tumor=1       
            
        
        image = pyvips.Image.new_from_file(self.cur_wsi_path[0])
        img_reg = pyvips.Region.new(image)
        if args.mask_path:
            mask_image  = pyvips.Image.new_from_file(self.cur_wsi_path[1])
            mask_reg = pyvips.Region.new(mask_image)
        
        numpy_batch_patch = []
        numpy_batch_mask  = []
        if os.path.isfile(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))):
            try:
                save_image = np.array(Image.open(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))))
            except:
                sleeptime=3
                print(f"waiting for save...")
                time.sleep(sleeptime)
                save_image = np.array(Image.open(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))))
                pass
                
        else:
            if args.mask_path:
                # copy image and mark tumor in black
                save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)
            else:
                save_image = self.rgb_image.copy()
        
        # for i in range(int(self.batch_size * (1 - self.tumor_ratio))):
        bc = random.choice(self.contours)
        msk = np.zeros(self.rgb_image.shape,np.uint8)
        cv2.drawContours(msk,[bc],-1,(255),-1)
        pixelpoints = np.transpose(np.nonzero(msk))
        
        b_x_start = bc[...,0].min() * self.mag_factor
        b_y_start = bc[...,1].min() * self.mag_factor
        b_x_end = bc[...,0].max() * self.mag_factor
        b_y_end = bc[...,1].max() * self.mag_factor
        h = b_y_end - b_y_start
        w = b_x_end - b_x_start
    
        patch = []
        
        while not len(patch):
            x_topleft = random.choice(pixelpoints)[1]* self.mag_factor
            y_topleft = random.choice(pixelpoints)[0]* self.mag_factor
            t1 = time.time()
            # if trying to fetch outside of image, retry
            try:
                patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                patch = np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3]
                _std = ImageStat.Stat(Image.fromarray(patch)).stddev
                # discard based on stddev
                if (sum(_std[:3]) / len(_std[:3])) < 15:
                    print("Discard based on stddev")
                    patch = []
            except:
                patch = []

        # im = self.wsi.read_region((x_topleft, y_topleft),0,(self.patch_size, self.patch_size))
        numpy_batch_patch.append(patch)

        if args.mask_path:
            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            numpy_batch_mask.append(np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8))
        
        print(f"{['Train' if self.train else 'Valid']} Sample {self.patch_size} x {self.patch_size} from contour = {h}" + f" by {w} in {time.time() -t1} seconds")

        # Draw the rectangles of sampled images on downsampled rgb
        save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
        save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                               (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                               (0,255,0), 2)
        
        # for i in range(int(self.batch_size * (self.tumor_ratio))):
        #     bb = random.choice(self.bounding_boxes_tumor)
        #     b_x_start = int(bb[0]) * self.mag_factor
        #     b_y_start = int(bb[1]) * self.mag_factor
        #     b_x_end = (int(bb[0]) + int(bb[2])) * self.mag_factor
        #     b_y_end = (int(bb[1]) + int(bb[3])) * self.mag_factor
        #     h = int(bb[2]) * self.mag_factor
        #     w = int(bb[3]) * self.mag_factor
            
        #     b_x_center = int((b_x_start + b_x_end) / 2)
        #     b_y_center = int((b_y_start + b_y_end) / 2)
        #     x_topleft = random.choice(range(b_x_center - int(0.5*self.patch_size),b_x_center))
        #     y_topleft = random.choice(range(b_y_center - int(0.5*self.patch_size),b_y_center))
            
        #     t1 = time.time()
        #     patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
        #     # im = self.wsi.read_region((x_topleft, y_topleft),0,(self.patch_size, self.patch_size))

        #     mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
        #     print(f"Sample {self.patch_size} x {self.patch_size} from bounding box = {h}" + f" by {w} in {time.time() -t1} seconds")
        #     numpy_batch_patch.append(np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3])
        #     numpy_batch_mask.append(np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8))
        #     # Draw the rectangles of sampled images on downsampled rgb
        #     save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
        #                                            (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
        #                                            (0,255,0), 2)


        Image.fromarray(save_image[...,:3]).save(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png')))

        
        # im = image.astype('uint8')
        # im = Image.fromarray(im)
        # im.save('test1.png')
        for image in numpy_batch_patch:
            image = imgtf.RGB2HSD(image/255.0).astype('float32')
        
        if args.mask_path:
            for mask in numpy_batch_mask:
                mask = mask
        
        image = make_dataset._transform(image,train=self.train)
        
        
        # im = image.permute(1,2,0)
        # im = im.cpu().detach().numpy()
        # im = im * 255
        # im = im.astype('uint8')
        # im = Image.fromarray(im)
        # im.save('test2.png')
        if args.mask_path:
            sample = (image ,mask, self.cur_wsi_path)
        else:
            sample = (image ,torch.zeros((1,self.patch_size,self.patch_size,1)), self.cur_wsi_path)


        return sample

# Dataset and hyperparameters
if args.data == 'celebahq':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 256:
        logger.info('Changing image size to 256.')
        args.imagesize = 256
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                reduce_bits,
                lambda x: add_noise(x, nvals=2**args.nbits),
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=False, transform=transforms.Compose([
                reduce_bits,
                lambda x: add_noise(x, nvals=2**args.nbits),
            ])
        ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
elif args.data == 'custom':
    im_dim = args.nclusters
    n_classes = args.nclusters
    init_layer = layers.LogitTransform(0.05)

    
    train_dataset = make_dataset(args,train=True)
    test_dataset  = make_dataset(args,train=False)
    # # Horovod: use DistributedSampler to partition the training data.
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize)
    # # Horovod: use DistributedSampler to partition the test data.
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batchsize)

if args.task in ['classification', 'hybrid','gmm']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification with {}'.format(args.data))
else:
    n_classes = 1

if rank00():
    logger.info('Dataset loaded.')
    logger.info('Creating model.')

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification=args.task in ['classification', 'hybrid'],
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)


# Custom
gmm = gmm(input_size,args,num_clusters=args.nclusters)
gmm.to(device)
model.to(device)
ema = utils.ExponentialMovingAverage(model)

def parallelize(model):
    return torch.nn.DataParallel(model)

if rank00():
    logger.info(model)
    logger.info('EMA: {}'.format(ema))

# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False


scheduler = None
params = [par for par in model.parameters()] + [par for par in gmm.parameters()]

# params = [par for par in gmm.parameters()]
if args.optimizer == 'adam':
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    if args.scheduler: scheduler = CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
elif args.optimizer == 'adamax':
    optimizer = optim.Adamax(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
        )
else:
    raise ValueError('Unknown optimizer {}'.format(args.optimizer))



# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none


optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Average)
# Horovod: broadcast parameters & optimizer state.

best_test_bpd = math.inf


if (args.resume is not None):

    if rank00(): logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    
    checkpt = torch.load(args.resume,map_location='cpu')
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    
    try:
        model.load_state_dict(state, strict=True)
    except ValueError("Model mismatch, check args.nclusters and args.nblocks"):
        sys.exit(1)
    
    # ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state



hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
hvd.join()



if rank00():
    logger.info(optimizer)

fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)



def compute_loss(x, model,gmm, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)


    if args.data == 'celebahq' or 'custom':
        nvals = 2**args.nbits
    else:
        nvals = 256

    # print(f"max {torch.max(x)} {torch.min(x)}")
    x, logpu = add_padding(x, nvals)

    if args.squeeze_first:
        x = squeeze_layer(x)
    
    if args.task == 'gmm' :
        D = x[:,0,...].unsqueeze(0).clone()
        D = rescale(D) # rescaling to [0,1]
        D = D.repeat(1, args.nclusters, 1, 1)
        z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
        

        z, delta_logp = z_logp
        
        # log p(z)
        # logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (args.imagesize * args.imagesize * (im_dim + args.padding)) - logpu
        bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp, params


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
ll_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)



def train(epoch,model,gmm):

    model.train()
    gmm.train()

    
    end = time.time()
    step = 0
    while step < args.epoch_steps:
        for idx, (x, y, z) in enumerate(train_loader):
            # break one iter early to avoid out of sync
    
            x = x.to(device)
            global_itr = epoch*args.epoch_steps + step
            update_lr(optimizer, global_itr)
            
            if rank00(): print(f'Step {step}')
            # Training procedure:
            # for each sample x:
            #   compute z = f(x)
            #   maximize log p(x) = log p(z) - log |det df/dx|
    
            beta = 1
            bpd, logits, logpz, neg_delta_logp, params = compute_loss(x, model,gmm, beta=beta)
    
            firmom, secmom = estimator_moments(model)
    
            bpd_meter.update(bpd.item())
            logpz_meter.update(logpz.item())
            deltalogp_meter.update(neg_delta_logp.item())
            firmom_meter.update(firmom)
            secmom_meter.update(secmom)
    
            # compute gradient and do SGD step
            # params = list(model.parameters())
            # grads  = [x.grad for x in params]
            # if rank00():
            #     print(params[-1])
            #     print(grads[-1])
                
            # for p in model.parameters():
            #     if utils.isnan(p.grad).any():
            #         p.grad[p.grad!=p.grad]=0.001
            #         p.grad[torch.abs(p.grad)==float('inf')]=0.001
                    
            loss = bpd
            loss.backward()
    
    
            if global_itr % args.update_freq == args.update_freq - 1:
                total_norm=0
                if args.update_freq >= 1:
                    with torch.no_grad():
                        grads = []
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad /= args.update_freq
                            grads.append(p.grad)
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            total_norm = total_norm ** (1. / 2)
                        
                # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
                grad_norm = total_norm
    
    
                if args.learn_p: compute_p_grads(model)
                
    
                optimizer.step()
                if rank00(): print("Optimizer.step() done")
                optimizer.zero_grad()
                
                update_lipschitz(model)
                ema.apply()
    
                gnorm_meter.update(grad_norm)
    
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0 and rank00():
                s = (
                    '\n\nEpoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                    'GradNorm {gnorm_meter.avg:.2f}'.format(
                        epoch, step, args.epoch_steps, batch_time=batch_time, gnorm_meter=gnorm_meter
                    )
                )
    
                if args.task in ['density', 'hybrid','gmm']:
                    s += (
                        f' | Bits/dim {bpd_meter.val}({bpd_meter.avg}) | '
                        # f' | params {[p.clone() for p in gmm.parameters().grad]}) | '
                        f'Logpz {logpz_meter.avg} | '
                        f'-DeltaLogp {deltalogp_meter.avg} | '
                        f'EstMoment ({firmom_meter.avg},{secmom_meter.avg})\n\n'
                        )
                
    
                logger.info(s)
            if global_itr % args.vis_freq == 0 and idx > 0 and rank00():
                visualize(epoch, model,gmm, idx, x, global_itr)
        
            del x
            torch.cuda.empty_cache()
            gc.collect()
            step += 1
    return

def savegamma(gamma,global_itr,pred=0):
    ClsLbl = np.argmax(gamma, axis=-1)
    ClsLbl = ClsLbl.astype('float32')
    
    ColorTable = [[255,0,0],[0,255,0],[0,0,255],[255,255,0], [0,255,255], [255,0,255]]
    colors = np.array(ColorTable, dtype='float32')
    Msk = np.tile(np.expand_dims(ClsLbl, axis=-1),(1,1,1,3))
    for k in range(0, args.nclusters):
        #                                       1 x 256 x 256 x 1                           1 x 3 
        ClrTmpl = np.einsum('anmd,df->anmf', np.expand_dims(np.ones_like(ClsLbl), axis=3), np.reshape(colors[k,...],[1,3]))
        # ClrTmpl = 1 x 256 x 256 x 3
        Msk = np.where(np.equal(Msk,k), ClrTmpl, Msk)
    
    im_gamma = Msk[0].astype('uint8')
    im_gamma = Image.fromarray(im_gamma)
    if pred == 0:
        im_gamma.save(os.path.join(args.save,'imgs',f'im_gamma_{global_itr}.png'))
    elif pred == 1:
        im_gamma.save(os.path.join(args.save,'imgs',f'im_pi_{global_itr}.png'))
    elif pred == 2:
        im_gamma.save(os.path.join(args.save,'imgs',f'im_recon_{global_itr}.png'))
    elif pred == 3:
        im_gamma.save(os.path.join(args.save,'imgs',f'im_fake_{global_itr}.png'))
        
    im_gamma.close()
    return
    
def validate(epoch, model,gmm, ema=None):
    """
    - Deploys the color normalization on test image dataset
    - Evaluates NMI / CV / SD
    """
    
        
    if rank00(): utils.makedirs(os.path.join(args.save, 'imgs')), print("Starting Validation")


    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model.eval()
    gmm.eval()

    mu_tmpl = 0
    std_tmpl = 0
    N = 0


    if rank00(): print(f"Deploying on templates...")
    for idx, (x, y, z_path) in enumerate(train_loader):
        # break one iter early to avoid out of sync
        if idx == len(train_loader) - 2: break
        t1 = time.time()
        
        x = x.to(device)
       
        ### TEMPLATES ###
        D = x[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

        
        mu, std, gamma =  params
        
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        gamma    = gamma.cpu().numpy() 
    
        
        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)

        N = N+1
        mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
        std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
        
        if idx % 50 == 0 and rank00(): print(f"Batch {idx} at { hvd.size()*args.batchsize / (time.time() - t1) } imgs / sec")
        
        if args.save_conv:
            # save images for transformation
            for ct, (img,path) in enumerate(zip(x,z_path)):
                im_tmpl = img.cpu().numpy()
                im_tmpl = np.swapaxes(im_tmpl,0,1)  
                im_tmpl = np.swapaxes(im_tmpl,1,-1)
                im_tmpl = imgtf.HSD2RGB_Numpy(im_tmpl)
                im_tmpl = (im_tmpl*255).astype('uint8')
                im_tmpl = Image.fromarray(im_tmpl)
                im_tmpl.save(os.path.join(args.save,'imgs',f'im_tmpl-{path.split("/")[-1]}-eval.png'))
                im_tmpl.close()
            
    if rank00(): print("Allreduce mu_tmpl / std_tmpl ...")
    mu_tmpl   = hvd.allreduce(torch.tensor(mu_tmpl).contiguous())
    std_tmpl  = hvd.allreduce(torch.tensor(std_tmpl).contiguous())
    if rank00(): print("Broadcast mu_tmpl / std_tmpl ...")
    hvd.broadcast(mu_tmpl,0)
    hvd.broadcast(std_tmpl,0)
    hvd.join()
    
    if rank00():
        print("Estimated Mu for template(s):")
        print(mu_tmpl)
          
        print("Estimated Sigma for template(s):")
        print(std_tmpl)
        
        del x
        torch.cuda.empty_cache()
        gc.collect()

         
    metrics = dict()
    for tc in range(1,args.nclusters+1):
        metrics[f'mean_{tc}'] = []
        metrics[f'median_{tc}']=[]
        metrics[f'perc_95_{tc}']=[]
        metrics[f'nmi_{tc}']=[]
        metrics[f'sd_{tc}']=[]
        metrics[f'cv_{tc}']=[]
    
    if rank00(): print(f"Predicting on templates...")
    for idx, (x_test, y_test, z_test) in enumerate(test_loader):
        # break one iter early to avoid out of sync
        if idx == len(test_loader) - 2: break
        t1 = time.time()
        x_test = x_test.to(device)
        

        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
        
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))

        
        mu, std, pi =  params

            
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        pi  = pi.cpu().numpy()
        
        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)
        
        X_hsd = np.swapaxes(x_test.cpu().numpy(),1,2)
        X_hsd = np.swapaxes(X_hsd,2,3)
        
        X_conv = imgtf.image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, args)

        ClsLbl = np.argmax(np.asarray(pi),axis=-1) + 1
        ClsLbl = ClsLbl.astype('int32')
        mean_rgb = np.mean(X_conv,axis=-1)
        for tc in range(1,args.nclusters+1):
            msk = torch.where(torch.tensor(ClsLbl) == tc , torch.tensor(1),torch.tensor(0))
            msk = [(i,msk.cpu().numpy()) for i, msk in enumerate(msk) if torch.max(msk).cpu().numpy()] # skip metric if no class labels are found
            if not len(list(msk)): continue
            # Take indices from valid msks and get mean_rgb at valid indices, then multiply with msk
            idces = [x[0] for x in msk]
            msk   = np.array([x[1] for x in msk])
            ma = mean_rgb[idces,...] * msk
            mean    = np.array([np.mean(ma[ma!=0]) for ma in ma])
            median  = np.array([np.median(ma[ma!=0]) for ma in ma])
            perc    = np.array([np.percentile(ma[ma!=0],95) for ma in ma])
            nmi = median / perc
            
            metrics['mean_'     +str(tc)].extend(list(mean))
            metrics['median_'   +str(tc)].extend(list(median))
            metrics['perc_95_'  +str(tc)].extend(list(perc))
            metrics['nmi_'      +str(tc)].extend(list(nmi))
            
        
        if args.save_conv:
            for ct, (img, path) in enumerate(zip(x_test,z_test)):
                im_test = img.cpu().numpy()
                im_test = np.swapaxes(im_test,0,1)
                im_test = np.swapaxes(im_test,1,-1)
                im_test = imgtf.HSD2RGB_Numpy(im_test)
                im_test = (im_test*255).astype('uint8')
                im_test = Image.fromarray(im_test)
                im_test.save(os.path.join(args.save,'imgs',f'im_test-{path.split("/")[-1]}-eval.png'))
                im_test.close()
            # for ct, img in enumerate(X_conv):
                im_conv = X_conv[ct].reshape(args.imagesize,args.imagesize,3)
                im_conv = Image.fromarray(im_conv)
                im_conv.save(os.path.join(args.save,'imgs',f'im_conv-{path.split("/")[-1]}-eval.png'))
                im_conv.close()
            
            # savegamma(pi,f"{idx}-eval",pred=1)
        
        
        if idx % 10 == 0 and rank00(): print(f"Batch {idx} at { hvd.size()*args.batchsize / (time.time() - t1) } imgs / sec")
    
    # average sd of nmi across tissue classes
    av_sd = []
    # average cv of nmi across tissue classes
    av_cv = []
    # total nmi across tissue classes
    tot_nmi = np.empty((0,0))
    
    
    for tc in range(1,args.nclusters+1):
        if len(metrics['mean_' + str(tc)]) == 0: continue
        nmi = hvd.allgather(torch.tensor(np.array(metrics['nmi_' + str(tc)])[...,None]))
        metrics[f'sd_' + str(tc)] = torch.std(nmi).cpu().numpy()
        metrics[f'cv_' + str(tc)] = torch.std(nmi).cpu().numpy() / torch.mean(nmi).cpu().numpy()
        if rank00():
            print(f'sd_' + str(tc)+':', metrics[f'sd_{tc}'])
            print(f'cv_' + str(tc)+':', metrics[f'cv_{tc}'])
        av_sd.append(metrics[f'sd_' + str(tc)])
        av_cv.append(metrics[f'cv_' + str(tc)])
        tot_nmi = np.append(tot_nmi,nmi)
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'Box Plot Eval {args.save.split("/")[-1]}')
    ax1.boxplot(tot_nmi)
    
    
    if rank00():
        plt.savefig(f'worker-{hvd.rank()}-{args.save.split("/")[-1]}-boxplot-eval.png')
        print(f"Average sd = {np.array(av_sd).mean()}")
        print(f"Average cv = {np.array(av_cv).mean()}")
        import csv
        file = open(f'worker-{hvd.rank()}-{args.save.split("/")[-1]}-metrics-eval.csv',"w")
        writer = csv.writer(file)
        for key, value in metrics.items():
            writer.writerow([key, value])
        
        
        file.close()
    

    # correct = 0
    # total = 0

    # start = time.time()
    # with torch.no_grad():
    #     for i, (x, y) in enumerate(tqdm(test_loader)):
    #         x = x.to(device)

    #         bpd, logits, _, _ = compute_loss(x, model)
    #         bpd_meter.update(bpd.item(), x.size(0))

    # val_time = time.time() - start

    # if ema is not None:
    #     ema.swap()
    # s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)
    # if args.task in ['classification', 'hybrid']:
    #     s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter.avg, 100 * correct / total)
    # logger.info(s)
    # return bpd_meter.avg
    
    return


def visualize(epoch, model, gmm, itr, real_imgs, global_itr):
    model.eval()
    gmm.eval()
    if rank00(): utils.makedirs(os.path.join(args.save, 'imgs')), print("Starting Visualisation")

    for x_test, y_test, z_test in test_loader:
        # x_test = x_test[0,...].unsqueeze(0)
        # y_test = y_test[0,...].unsqueeze(0)
        x_test = x_test.to(device)
        ### TEMPLATES ###
        D = real_imgs[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        x = real_imgs
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
            
            
            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x.permute(0,2,3,1))

    
        mu_tmpl, std_tmpl, gamma =  params
        mu_tmpl  = mu_tmpl.cpu().numpy()
        std_tmpl = std_tmpl.cpu().numpy()
        gamma    = gamma.cpu().numpy() 
    
        mu_tmpl  = mu_tmpl[...,np.newaxis]
        std_tmpl = std_tmpl[...,np.newaxis]
        
        mu_tmpl = np.swapaxes(mu_tmpl,0,1) # (3,4,1) -> (4,3,1)
        mu_tmpl = np.swapaxes(mu_tmpl,1,2) # (4,3,1) -> (4,1,3)
        std_tmpl = np.swapaxes(std_tmpl,0,1) # (3,4,1) -> (4,3,1)
        std_tmpl = np.swapaxes(std_tmpl,1,2) # (4,3,1) -> (4,1,3)
        
            
        ### DEPLOY ###
        D = x_test[:,0,...].unsqueeze(1).clone()
        D = rescale(D) # Scale to [0,1] interval
        D = D.repeat(1, args.nclusters, 1, 1)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                z_logp = model.module(D.view(-1, *input_size[1:]), 0, classify=False)
            else:
                z_logp = model(D.view(-1, *input_size[1:]), 0, classify=False)
            
                # recon = model(model(D.view(-1, *input_size[1:])), inverse=True).view(-1, *input_size[1:])
                # fake_imgs = model(fixed_z, inverse=True).view(-1, *input_size[1:])

            z, delta_logp = z_logp
            if isinstance(gmm,torch.nn.DataParallel):
                logpz, params = gmm.module(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))
            else:
                logpz, params = gmm(z.view(-1,args.nclusters,args.imagesize,args.imagesize), x_test.permute(0,2,3,1))

        mu, std, pi =  params
        mu  = mu.cpu().numpy()
        std = std.cpu().numpy()
        pi  = pi.cpu().numpy()
        # recon  = np.swapaxes(np.swapaxes(recon.cpu().numpy(),1,2),2,-1)
        # fake_imgs  = np.swapaxes(np.swapaxes(fake_imgs.cpu().numpy(),1,2),2,-1)

        mu  = mu[...,np.newaxis]
        std = std[...,np.newaxis]
        
        mu = np.swapaxes(mu,0,1) # (3,4,1) -> (4,3,1)
        mu = np.swapaxes(mu,1,2) # (4,3,1) -> (4,1,3)
        std = np.swapaxes(std,0,1) # (3,4,1) -> (4,3,1)
        std = np.swapaxes(std,1,2) # (4,3,1) -> (4,1,3)
        
    
        X_hsd = np.swapaxes(x_test.cpu().numpy(),1,2)
        X_hsd = np.swapaxes(X_hsd,2,3)

        X_conv       = imgtf.image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, args)
        # X_conv_recon = imgtf.image_dist_transform(X_hsd, mu, std, recon, mu_tmpl, std_tmpl, args)
        # save a random image from the batch
        im_no = random.randint(0,args.val_batchsize-1) 
        im_tmpl = real_imgs[im_no,...].cpu().numpy()
        im_tmpl = np.swapaxes(im_tmpl,0,1)
        im_tmpl = np.swapaxes(im_tmpl,1,-1)
        im_tmpl = imgtf.HSD2RGB_Numpy(im_tmpl)
        im_tmpl = (im_tmpl*255).astype('uint8')
        im_tmpl = Image.fromarray(im_tmpl)
        im_tmpl.save(os.path.join(args.save,'imgs',f'im_tmpl_{global_itr}.png'))
        
        im_test = x_test[im_no,...].cpu().numpy()
        im_test = np.swapaxes(im_test,0,1)
        im_test = np.swapaxes(im_test,1,-1)
        im_test = imgtf.HSD2RGB_Numpy(im_test)
        im_test = (im_test*255).astype('uint8')
        im_test = Image.fromarray(im_test)
        im_test.save(os.path.join(args.save,'imgs',f'im_test_{global_itr}.png'))
        
        im_D = D[0,0,...].cpu().numpy()
        im_D = (im_D*255).astype('uint8')
        im_D = Image.fromarray(im_D,'L')
        im_D.save(os.path.join(args.save,'imgs',f'im_D_{global_itr}.png'))
        if args.val_batchsize>1:
            im_conv = X_conv[im_no,...].reshape(args.imagesize,args.imagesize,3)
        else:
            im_conv = X_conv.reshape(args.imagesize,args.imagesize,3)
        im_conv = Image.fromarray(im_conv)
        im_conv.save(os.path.join(args.save,'imgs',f'im_conv_{global_itr}.png'))
        
        # im_conv_recon = np.squeeze(X_conv_recon)[im_no,...].reshape(args.imagesize,args.imagesize,3)
        # im_conv_recon = Image.fromarray(im_conv_recon)
        # im_conv_recon.save(os.path.join(args.save,'imgs',f'im_conv_recon_{global_itr}.png'))
        
        # gamma
        savegamma(gamma,global_itr,pred=0)
        
        # pi
        savegamma(pi,global_itr,pred=1)
        # reconstructed images
        # savegamma(recon,global_itr,pred=2)
        
        # reconstructed images
        # savegamma(fake_imgs,global_itr,pred=3)
        
        model.train()
        gmm.train()
        return


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'


def main():
    global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []

    if args.resume:
        validate(args.begin_epoch - 1, model,gmm)
        sys.exit(0)
        
    for epoch in range(args.begin_epoch, args.nepochs):
        
        if rank00():
            logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))


        
        train(epoch, model, gmm)
        lipschitz_constants.append(get_lipschitz_constants(model))
        ords.append(get_ords(model))
        if rank00():
            logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
            logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        # if args.ema_val:
        #     validate(epoch, model,gmm,ema)
        # else:
        #     validate(epoch, model,gmm)
        

        if args.scheduler and scheduler is not None:
            scheduler.step()

        
        if rank00() and epoch % args.save_every == 0 and epoch > 0:
            print("Saving model...")
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                # 'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)
    
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                # 'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models', f'most_recent_{hvd.size()}_workers.pth'))
        
        # validate(args.begin_epoch - 1, model,gmm, ema)
            


if __name__ == '__main__':
    main()
