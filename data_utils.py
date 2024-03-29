import torch
import cv2
import numpy as np
from glob import glob
import difflib
import os
import horovod.torch as hvd
from PIL import Image,ImageStat
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
import pyvips
import random
import torchvision.transforms as transforms
from sklearn.utils import shuffle
import lib.image_transforms as imgtf
import itertools
import xml.etree.ElementTree as ET
import pdb
import random 

# PyVips Conversion
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

    

class make_dataset():
    """
    - This sampler samples patches from whole slide images  in several formats, from
    which it samples the patch on the WSI and the same patch on the WSI mask.
    
    !! This Sampler needs to be used with OpenSlide and PyVips library installed
    in the backend, see https://github.com/sara-nl/SURF-deeplab/blob/master/README.md
    
    - Furthermore it needs to glob over directories that have the following structure:
        
    ---`opts.slide_path`/
                        <WSI_001>.`opts.slide_format`
                        <WSI_002>.`opts.slide_format` 
                        ...
                        
    ---`opts.label_path`/
                        <WSI_Mask_001>.`opts.label_format`
                        <WSI_Mask_002>.`opts.label_format` 
                        ...
    
    
    !! Label and WSI's are matched on string similarity (https://docs.python.org/3/library/difflib.html -> difflib.get_close_matches() )
    
    - It samples a batch according to `opts.batch_size`, with the batch 
    consisting of  `(patches / 255).astype('float32'), (masks / 255).astype('float32')`
    that contain tumor and non - tumor, based on 
    `opts.batch_tumor_ratio` \in [0,1] (rounded to ints)
    
    - It samples out of contours constructed with OpenCV thresholding
    
    - Furthermore it contains a hard-coded standard deviation threshold, which 
    can discard patches if not above some stddev. This is to avoid sampling
    patches that are background. From experience on CAMELYON16/17 this works
    as intended, no guarantees are given for other datasets

    - When mode == 'train':
        > Trainer function is used for sampling patches(masks) from opts.slide(label)_path
    
    - When mode == 'validation':
        > Tester function is used for sampling patches(masks) from opts.slide(label)_path * (1 - opt.val_split)
        if no opts.valid_slide(label)_path else use opts.valid_slide(label)_path
    
    - When mode == 'test':
        > Tester function is used for sampling patches from opts.test_path 
        if no opts.test_path use opts.valid_slide(label)_path
        
   A configuration object is needed with the following configs:
   - config.slide_path          -> str
   - config.label_path          -> str
   - config.slide_format        -> str
   - config.label_format        -> str
   - config.valid_slide_path    -> str*
   - config.valid_label_path    -> str*
   - config.test_path           -> str*
   - config.batch_size          -> int
   - config.bb_downsample       -> int
   - config.img_size            -> int
   - config.batch_tumor_ratio   -> float [0,1]
   - config.log_dir             -> str
   - config.evaluate            -> bool
   - config.verbose             -> str
   - config.steps_per_epoch     -> str

    * = optional

   >>>> Example:

    train_sampler = SurfSampler(config,mode='train')
    image, mask   = train_sampler.__getitem__(0)

    """
    def __init__(self, opts, mode='train'):

    
        try:
            super(tf.keras.utils.Sequence).__init__()
        except:
            try:
                super(torch.utils.data.Dataset).__init__()
            except Exception as e:
                print(e,"\nImportError, Make sure [import torch] or [import tensorflow as tf]")


        self.mode = mode.lower()
        
        # Get list of paths
        slides = sorted(glob(os.path.join(opts.slide_path,f'*.{opts.slide_format}')))
        labels = sorted(glob(os.path.join(opts.label_path,f'*.{opts.label_format}')))
        
        
        # Match labels to slides (all slides must have labels)
        self.train_paths = shuffle([(difflib.get_close_matches(label.split('/')[-1].split('.')[-2],slides,n=1,cutoff=0.1)[0],label) for label in labels])
        
        
        # Custom path removal for Camelyon 17
        if opts.slide_path.find('CAMELYON17') > 0:
            _del = []
            for data in self.train_paths:
                if data[0].split('/')[-1].split('.')[-2] != data[1].split('/')[-1].split('.')[-2]:
                    _del.append(data)
            
            self.train_paths = [data for data in self.train_paths if data not in _del]
        
        if hvd.rank() == 0 : print(f"\nFound {len(self.train_paths)} slides")
        
        # Get validation data
        if opts.valid_slide_path:
            valid_slides = glob(os.path.join(opts.valid_slide_path,f'*.{opts.slide_format}'))
            valid_labels = glob(os.path.join(opts.valid_label_path,f'*.{opts.label_format}'))
            
            # Match labels to slides (all slides must have labels)
            self.valid_paths = [(difflib.get_close_matches(label.split('/')[-1],valid_slides,n=1,cutoff=0.1)[0],label) for label in valid_labels]
        else:
            val_split = int(len(self.train_paths) * (1-opts.val_split))
            val_split = min(len(self.train_paths)-1,val_split)
            self.valid_paths = self.train_paths[val_split:]
            self.train_paths = self.train_paths[:val_split]
            
        # Get test data
        if opts.test_path:
            self.test_paths = glob(os.path.join(opts.test_path,f'*.{opts.slide_format}'))
        elif mode == 'test' and not opts.test_path:
            self.test_paths = self.valid_paths
        else:
            self.test_paths = None

        if hvd.rank() == 0 : 
            print(f"\nWith {len(self.train_paths)} slides")
            print(f"and {len(self.valid_paths)} validation slides\n")
            if opts.test_path:
                print(f"and {len(self.test_paths)} test slides\n")
        
        self.opts           = opts    
        self.batch_size     = 1
        self.contours_train = []
        self.contours_valid = []
        self.contours_test  = []
        self.contours_tumor = []
        self.pixelpoints    = []
        self.save_data      = []
        self.level_used     = opts.bb_downsample
        self.mag_factor     = pow(2, self.level_used)
        self.patch_size     = opts.img_size
        self.tumor_ratio    = opts.batch_tumor_ratio
        self.log_image_path = opts.log_dir
        self.slide_format   = opts.slide_format
        self.evaluate       = opts.evaluate
        self.cnt            = 0
        self.verbose        = opts.verbose
        self.steps_per_epoch= opts.steps_per_epoch
        
        

         # Make sure that every process has at least 1 WSI
        if opts.test_path:
             self.test_paths  = self.test_paths*hvd.size()
             assert hvd.size() <= len(self.test_paths), f"WARNING: {hvd.size()} workers will share {len(self.test_paths)} {mode} images"
             testims     = len(self.test_paths)
             test_per_worker     = testims // hvd.size()
             print(f"Worker {hvd.rank()}: len = {len(self.test_paths)}")
             self.test_paths = self.test_paths[hvd.rank()*test_per_worker:(hvd.rank()+1)*test_per_worker]
             print(f"Worker {hvd.rank()}: AFTER len = {len(self.test_paths)}")
        else:
            self.valid_paths = self.valid_paths*hvd.size()*15
            assert hvd.size() <= len(self.valid_paths), f"WARNING: {hvd.size()} workers will share {len(self.valid_paths)} {mode} images"
            validims    = len(self.valid_paths)
            valid_per_worker    = validims // hvd.size()
            print(f"Worker {hvd.rank()}: len = {len(self.valid_paths)} Validation")
            self.valid_paths = self.valid_paths[hvd.rank()*valid_per_worker:(hvd.rank()+1)*valid_per_worker]
            print(f"Worker {hvd.rank()}: AFTER len = {len(self.valid_paths)} Validation")

    def reduce_bits(self,x):
        if self.opts.nbits < 8:
            x = x * 255
            x = torch.floor(x / 2**(8 - self.opts.nbits))
            x = x / 2**self.opts.nbits
        return x


    def add_noise(self,x, nvals=256):
        """
        [0, 1] -> [0, nvals] -> add noise -> [0, 1]
        """
        if self.opts.add_noise:
            noise = x.new().resize_as_(x).uniform_()
            x = x * (nvals - 1) + noise
            x = x / nvals
        return x
    
    def rescale(self,x):
        """
        Parameters
        ----------
        tensor : Pytorch tensor
            Tensor to be rescaled to [0,1] interval.
    
        Returns
        -------
        Rescaled tensor.
    
        """
        x -= x.min()
        x /= x.max()
        return x
        
    def transform(self,image):
        
        if self.mode =='train' :
            return transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                lambda x: make_dataset.reduce_bits(self,x),
                                lambda x: make_dataset.add_noise(self,x, nvals=2**self.opts.nbits),
                                # Do not rescale, it will not make RGB2HSD work see lib.image_tranforms
                                # lambda x: make_dataset.rescale(self,x),
                            ])(image)
        else:
            return transforms.Compose([
                                transforms.ToTensor(),
                                lambda x: make_dataset.reduce_bits(self,x),
                                lambda x: make_dataset.add_noise(self,x, nvals=2**self.opts.nbits),
                                # Do not rescale, it will not make RGB2HSD work see lib.image_tranforms
                                # lambda x: make_dataset.rescale(self,x),
                            ])(image)
        
        
    def __len__(self):
        return self.steps_per_epoch

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
    
    def trainer(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image):
                    
        tumor_count = 0
        tumor_patches = round(self.batch_size * self.tumor_ratio)
        for i in range(int(self.batch_size)): 
            patch = []
            k=0
            while not len(patch):
                if tumor_count < tumor_patches:
                    bc = random.choice(self.contours_tumor)
                    tumor_count += 1
                else:
                    bc = random.choice(self.contours)
                    
                msk = np.zeros(self.rgb_image.shape,np.uint8)
                # First get coords from contours of tumor, after tumor get negative coords 
                cv2.drawContours(msk,[bc],-1,(255),-1)
                pixelpoints = np.transpose(np.nonzero(msk)) * self.mag_factor
                                     
                b_x_start = bc[...,0].min() * self.mag_factor
                b_y_start = bc[...,1].min() * self.mag_factor
                b_x_end = bc[...,0].max() * self.mag_factor
                b_y_end = bc[...,1].max() * self.mag_factor
                h = b_y_end - b_y_start
                w = b_x_end - b_x_start
                pixelcoords = random.choice(pixelpoints)
                x_topleft = pixelcoords[1] 
                y_topleft = pixelcoords[0] 
                
                # if trying to fetch outside of image, retry
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3]
                    _std = ImageStat.Stat(Image.fromarray(patch)).stddev
                    
                    k+=1
                    # discard based on stddev
                    if k < 10: 
                        if (sum(_std[:3]) / len(_std[:3])) < 15:
                            if self.verbose == 'debug':
                                print("Discard based on stddev")
                                patch = []
                    
                    msk_downsample = 1
                    mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size//msk_downsample, self.patch_size//msk_downsample)
                    mask  = np.ndarray((self.patch_size//msk_downsample,self.patch_size//msk_downsample,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
                
                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = np.random.normal(size=(self.patch_size,self.patch_size,3))
                    mask  = np.random.normal(size=(self.patch_size,self.patch_size,1))
                

            numpy_batch_patch.append(patch)
            numpy_batch_mask.append(mask)
            x,y,imsize = x_topleft, y_topleft, self.patch_size
            coords = list(itertools.product(list(range(y,y+imsize)),list(range(x,x+imsize))))

            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                                        (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                                        (255,255,255), -1)
            except:
                pass
            
            self.save_data.append(({   'patch'      : patch,
                                       'image'      : save_image,
                                       'file_name'  : self.cur_wsi_path[0],
                                       'coords'     : coords,
                                       'mask'       : mask,
                                       'tumor'      : 1}))
        try:
            Image.fromarray(save_image[...,:3]).save(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png')))
        except:
            pass
        

        return np.array(numpy_batch_patch),np.array(numpy_batch_mask)
    
    
    def tester(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image):
        
        tumor_count = 0
        tumor_patches = round(self.batch_size * self.tumor_ratio)
        for i in range(int(self.batch_size)):
            patch = []
            while not len(patch):
                try:
                    if self.mode == 'validation' and not self.opts.evaluate:
                        if tumor_count < tumor_patches:
                            bc = random.choice(self.contours_tumor)
                            tumor_count += 1
                        else:
                            bc = self.contours[self.cnt]
                    else:
                        bc = self.contours[self.cnt]
                except:
                    print(f"WARNING: WSI {self.cur_wsi_path[0]} has no contours")
                    continue                
                
                msk = np.zeros(self.rgb_image.shape,np.uint8)
                x_topleft,y_topleft,width,height = cv2.boundingRect(bc)
                cv2.drawContours(msk,[bc],-1,(255),-1)
                # First gather all posssible pixelpoints, then, drop past_coords
                if not len(self.pixelpoints):
                    self.pixelpoints = np.transpose(np.nonzero(msk))
                    self.pixelpoints = self.pixelpoints[...,:2] * self.mag_factor
                    

                b_x_start = bc[...,0].min() * self.mag_factor
                b_y_start = bc[...,1].min() * self.mag_factor
                b_x_end = bc[...,0].max() * self.mag_factor
                b_y_end = bc[...,1].max() * self.mag_factor
                h = b_y_end - b_y_start
                w = b_x_end - b_x_start
                
                pixelcoords = random.choice(self.pixelpoints)
                x_topleft = pixelcoords[1]
                y_topleft = pixelcoords[0]
                
                
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3]
                    msk_downsample = 1
                    if not self.mode == 'test':
                        # mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                        mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size//msk_downsample, self.patch_size//msk_downsample)
                        # mask  = np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
                        mask  = np.ndarray((self.patch_size//msk_downsample,self.patch_size//msk_downsample,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
                    else:
                        mask=[]
                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = np.random.normal(size=(self.patch_size,self.patch_size,3))
                    if not self.mode == 'test':
                        mask  = np.random.normal(size=(self.patch_size,self.patch_size,1))
                    else:
                        mask = []
                    continue
                    
            # if hvd.rank() ==0 : print(f"\n\nTest Sample {self.patch_size} x {self.patch_size} from ROI = {h}" + f" by {w} in {time.time() -t1} seconds\n\n")
            numpy_batch_patch.append(patch)            
            numpy_batch_mask.append(mask)
            
            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                                        (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                                        (255,255,255), -1)
            except:
                pass
            
            x,y,imsize = x_topleft, y_topleft, self.patch_size
            # Get Cartesian product so all patch coordinates are dropped from pixelpoints
            coords = list(itertools.product(list(range(y,y+imsize)),list(range(x,x+imsize))))
            
            self.save_data = [{     'patch'      : patch,
                                    'image'      : save_image,
                                    'file_name'  : self.cur_wsi_path[0],
                                    'coords'     : coords,
                                    'mask'       : mask,
                                    'tumor'      : 1*(np.count_nonzero(mask) > 0)}]
            
            # Remove past coordinates from possible pixelpoints
            mask_pixel_in_past_coords=np.isin(self.pixelpoints,coords,invert=True)
            pixel_not_in_past_coords=np.nonzero(mask_pixel_in_past_coords)
            row,co = pixel_not_in_past_coords[0],pixel_not_in_past_coords[1]
            row=row[co>0]
            self.pixelpoints = self.pixelpoints[row,:]
        
        print(f"\n\nTest sampling at ROI {self.cnt+1} / {len(self.contours)} of {self.cur_wsi_path} with ~ {len(self.pixelpoints) // (self.batch_size*self.patch_size)} iter to go.\n\n")
            
        # If past all patches of contour, get next contour
        if len(self.pixelpoints) <= self.patch_size:
        # if 1: # for debugging
            self.cnt +=1
            self.pixelpoints = []
            
            if self.cnt == len(self.contours): 
                self.wsi_idx +=1
                self.cnt = 0
             
        try:
            Image.fromarray(save_image[...,:3]).save(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png')))
        except:
            pass

        
        return np.array(numpy_batch_patch),np.array(numpy_batch_mask) 
    
        
    def parse_xml(self,label=None):
        """
            make the list of contour from xml(annotation file)
            input (CAMELYON17):
                
        <?xml version="1.0"?>
        <ASAP_Annotations>
        	<Annotations>
        		<Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="metastases" Color="#F4FA58">
        			<Coordinates>
        				<Coordinate Order="0" X="12711.2998" Y="88778.1016" /> 
                        .
                        .
                        .
        			</Coordinates>
        		</Annotation>
        	</Annotations>
        	<AnnotationGroups>
        		<Group Name="metastases" PartOfGroup="None" Color="#ff0000">
        			<Attributes />
        		</Group>
        	</AnnotationGroups>
        </ASAP_Annotations>
            
            fn_xml = file name of xml file
            downsample = desired resolution
            var:
            li_li_point = list of tumors
            li_point = the coordinates([x,y]) of a tumor
            return  list of list (2D array list)
        """

        li_li_point = []
        tree = ET.parse(label)
        
        for ASAP_Annotations in tree.getiterator():
            for i_1, Annotations in enumerate(ASAP_Annotations):
                for i_2, Annotation in enumerate(Annotations):
                    for i_3, Coordinates in enumerate(Annotation):
                        li_point = []
                        for i_4, Coordinate in enumerate(Coordinates):
                            x_0 = float(Coordinate.attrib['X'])
                            y_0 = float(Coordinate.attrib['Y'])
                            li_point.append((x_0, y_0))
                        if len(li_point):
                            li_li_point.append(li_point)

       
        # Make opencv contours
        contours = []
        for li_point in li_li_point:
            li_point_int = [[int(round(point[0])), int(round(point[1]))] for point in li_point]
            contour = np.array(li_point_int, dtype=np.int32)
            contours.append(contour)
        
        # Make lvl 0 mask, with width - height dimensionality

        mask = np.zeros(tuple(reversed(self.wsi.dimensions)),dtype=np.uint8)
        
        for idx,contour in enumerate(contours):
            cv2.fillPoly(mask, pts =[contour], color=(255))
            
        return mask
    
    def __getitem__(self,idx):
        
        # Every new iteration, new sample
        
        cnt = 0
        
        if self.mode == 'train':
            idx = random.randint(0,len(self.train_paths)-1)
            while not self.contours_train or not self.contours_tumor:
                try:
                    if cnt > 5: idx = random.randint(0,len(self.train_paths)-1)
                    self.cur_wsi_path = self.train_paths[idx]
                    if hvd.rank() ==0  and self.verbose == 'debug': print(f"Opening {self.cur_wsi_path}...")
                    
                    
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                    
                    if self.opts.label_format.find('xml') > -1:
                        self.mask =  make_dataset.parse_xml(self,label=self.cur_wsi_path[1])
                    else:
                        self.mask = OpenSlide(self.cur_wsi_path[1])
                    
                    self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.rgb_image = np.array(self.rgb_image_pil)
                    
                    if self.opts.label_format.find('xml') > -1:
                        self.mask_image = cv2.resize(self.mask,self.wsi.level_dimensions[self.level_used])[...,None]
                    else:
                        self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                        self.mask_image = np.array(self.mask_pil)
                        
                    self.contours_train = self.get_bb()
                    self.contours = self.contours_train
                    
                    # Get contours of tumor, if not tumor, patch is negative
                    contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        self.contours_tumor = contours
                    else:
                        self.contours_tumor = self.contours
                        
                    cnt += 1
                except Exception as e:
                    print(f"{e}, at {self.cur_wsi_path[0]}")
                    cnt += 1
                    try:
                        self.wsi.close()
                        self.mask.close()
                    except:
                        pass
                    pass
                
        elif self.mode == 'validation':
            while not self.contours_valid or not self.contours_tumor:
                try:
                    if cnt > 5: idx +=1
                    if idx < len(self.valid_paths):
                        self.wsi_idx = idx
                    else:
                        cnt += 1
                        continue
                    # Drop past WSI's
                    self.valid_paths_new = self.valid_paths[self.wsi_idx:]
                    
                    self.cur_wsi_path = self.valid_paths_new[0]
                    if hvd.rank() ==0  and self.verbose == 'debug': print(f"Opening {self.cur_wsi_path}...")
                    
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                    if self.opts.label_format.find('xml') > -1:
                        self.mask =  make_dataset.parse_xml(self,label=self.cur_wsi_path[1])
                    else:
                        self.mask = OpenSlide(self.cur_wsi_path[1])
                    
                    self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.rgb_image = np.array(self.rgb_image_pil)
                    if self.opts.label_format.find('xml') > -1:
                        self.mask_image = cv2.resize(self.mask,self.wsi.level_dimensions[self.level_used])[...,None]
                    else:
                        self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                        self.mask_image = np.array(self.mask_pil)
                    
                    self.contours_valid = self.get_bb()
                    if not self.contours_valid: self.valid_paths.remove(self.cur_wsi_path)
                    self.contours = self.contours_valid
                    
                    # Get contours of tumor, if not tumor, patch is negative
                    contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        self.contours_tumor = contours
                    else:
                        self.contours_tumor = self.contours
                    cnt += 1
                    
                except Exception as e:
                    print(f"{e}, at {self.cur_wsi_path[0]}")
                    cnt += 1
                    try:
                        self.wsi.close()
                        self.mask.close()
                    except:
                        pass
                    pass
                
                    
        elif self.mode == 'test':

            while not self.contours_test:
                try:
                    if cnt > 5: idx +=1
                    if idx < len(self.test_paths):
                        self.wsi_idx = idx
                    else:
                        continue
                    
                    # Drop past WSI's

                    self.test_paths_new = self.test_paths[self.wsi_idx:]
                    self.cur_wsi_path = [self.test_paths_new[0]]

                    
                    # If a validation tuple of (image,mask), get image
                    if isinstance(self.cur_wsi_path[0],tuple):
                        self.cur_wsi_path = self.cur_wsi_path[0]
                        
                    if hvd.rank() == 0 and self.verbose == 'debug': print(f"Opening {self.cur_wsi_path}...")
                    
                    # OpenSlide and get contours of ROI
                    self.wsi  = OpenSlide(self.cur_wsi_path[0])
                    self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.rgb_image = np.array(self.rgb_image_pil)
                    self.contours_test = self.get_bb()
                    if not self.contours_test: self.test_paths.remove(self.cur_wsi_path)
                    self.contours = self.contours_test
                    cnt += 1
                except Exception as e:
                    print(f"{e}, at {self.cur_wsi_path[0]}")
                    cnt += 1
                    try:
                        self.wsi.close()
                    except:
                        pass
                    pass
        else:
            raise ValueError("Please submit suitable Sampler mode")
                
        
        image = pyvips.Image.new_from_file(self.cur_wsi_path[0])
        if not self.mode == 'test':
            if self.opts.label_format.find('xml') > -1:
                mask_image  = pyvips.Image.new_from_memory(self.mask,self.mask.shape[1],self.mask.shape[0],1,dtype_to_format[str(self.mask.dtype)])
            else:
                mask_image  = pyvips.Image.new_from_file(self.cur_wsi_path[1])
            mask_reg = pyvips.Region.new(mask_image)
        
        img_reg = pyvips.Region.new(image)
        
        
        numpy_batch_patch = []
        numpy_batch_mask  = []
        if os.path.isfile(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))):   
            try:
                save_image = np.array(Image.open(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))))
            except:
                try:
                    save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)
                except:
                    save_image = self.rgb_image.copy()[...,:3]
        else:
            try:
                # copy image and mark tumor in black
                save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)
            except:
                save_image = self.rgb_image.copy()[...,:3]
        
        if self.mode == 'test':
                mask_reg = None
                mask_image = None
        

        if self.mode == 'test' or self.mode == 'validation':
            patches, masks = make_dataset.tester(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image)
        else:
            patches, masks = make_dataset.trainer(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image)
            self.save_data = []
    
        self.wsi.close()
        if hasattr(self,'mask'):
            del self.mask
        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []
        self.contours_test  = []
        
        _lst = []
        for idx, image in enumerate(patches):
            # image in [0,255]
            HSD = imgtf.RGB2HSD(image/1.0).astype('float32')
            _lst.append(HSD)
        
        patches = np.array(_lst)
        patches,masks = patches[0],masks[0]
        patches = make_dataset.transform(self,patches)
        # def set_shapes(self,image, label):
        #     image,label = (image / 255), (label / 255) #(image / 255).astype('float32'), (label / 255).astype('float32')
        #     image.set_shape((self.opts.image_size,self.opts.image_size,3))
        #     label.set_shape((self.opts.image_size,self.opts.image_size,1))
        #     return image, label
        
        # dataset = tf.data.Dataset.from_tensor_slices((np.array(numpy_batch_patch),np.array(numpy_batch_mask)))
        # dataset = dataset.map(lambda x,y: set_shapes(self,x,y),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # return dataset
        # print(f"Got item with shape {patches.shape},{masks.shape}")
        return (patches).to(torch.float32), (torch.tensor(masks)/ 255.).to(torch.float32)
