import tensorflow as tf
import numpy as np
import os, fnmatch
import pdb
import torch

def RGB2HSD_GPU(X):
    X = X.permute(0,2,3,1)
    eps = torch.tensor([np.finfo(float).eps]).cuda()
    X = torch.where(X==0,eps,X)
    
    OD = -torch.log(X / 1.0)
    D  = torch.mean(OD,X.ndim - 1)
    D  = torch.where(D == 0.0, eps, D)
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (torch.sqrt(torch.tensor([3.0])).cuda()*D)
                                      
    D  = D.unsqueeze(3)
    cx = cx.unsqueeze(3)
    cy = cy.unsqueeze(3)
            
    X_HSD = torch.cat((D,cx,cy),3)
    
    return X_HSD.permute(0,3,1,2)


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,-1)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,0] / (D) - 1.0
    cy = (OD[:,:,1]-OD[:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,-1)
    cx = np.expand_dims(cx,-1)
    cy = np.expand_dims(cy,-1)
            
    X_HSD = np.concatenate((D,cx,cy),-1)
    return X_HSD
    
def HSD2RGB(X_HSD):
    
    X_HSD_0, X_HSD_1, X_HSD_2  = tf.split(X_HSD, [1,1,1], axis=3)
    D_R = (X_HSD_1+1) * X_HSD_0
    D_G = 0.5*X_HSD_0*(2-X_HSD_1 + tf.sqrt(tf.constant(3.0))*X_HSD_2)
    D_B = 0.5*X_HSD_0*(2-X_HSD_1 - tf.sqrt(tf.constant(3.0))*X_HSD_2)
    
    X_OD = tf.concat([D_R,D_G,D_B],3)
    X_RGB = 1.0 * tf.exp(-X_OD)
    return X_RGB   
    
def HSD2RGB_Numpy(X_HSD):
    
    X_HSD_0 = X_HSD[...,0]
    X_HSD_1 = X_HSD[...,1]
    X_HSD_2 = X_HSD[...,2]
    D_R = np.expand_dims(np.multiply(X_HSD_1+1 , X_HSD_0), -1)
    D_G = np.expand_dims(np.multiply(0.5*X_HSD_0, 2-X_HSD_1 + np.sqrt(3.0)*X_HSD_2), -1)
    D_B = np.expand_dims(np.multiply(0.5*X_HSD_0, 2-X_HSD_1 - np.sqrt(3.0)*X_HSD_2), -1)
    X_OD = np.concatenate((D_R,D_G,D_B), axis=-1)
    X_RGB = 1.0 * np.exp(-X_OD)
    return X_RGB         
    

def image_dist_transform(img_hsd, mu, std, gamma, mu_tmpl, std_tmpl, args):

    batch_size = args.batchsize
    img_norm = np.empty((batch_size,args.imagesize, args.imagesize, 3, args.nclusters))
    mu  = np.reshape(mu, [mu.shape[0] ,batch_size,1,1,3])
    std = np.reshape(std,[std.shape[0],batch_size,1,1,3])
    mu_tmpl  = np.reshape(mu_tmpl, [mu_tmpl.shape[0] ,batch_size,1,1,3])
    std_tmpl = np.reshape(std_tmpl,[std_tmpl.shape[0],batch_size,1,1,3])
    for c in range(0, args.nclusters):
        img_normalized = np.divide(np.subtract(np.squeeze(img_hsd), mu[c, ...]), std[c, ...])
        img_univar = np.add(np.multiply(img_normalized, std_tmpl[c, ...]), mu_tmpl[c, ...])
        # img_univar = np.add(np.zeros_like(img_norm), mu[c,...])
        img_norm[..., c] = np.multiply(img_univar, np.tile(np.expand_dims(np.squeeze(gamma[..., c]), axis=-1), (1, 1, 3)))

    
    img_norm = np.sum(img_norm, axis=-1)
    # Apply the triangular restriction to cxcy plane in HSD color coordinates
    img_norm = np.split(img_norm, 3, axis=-1)
    
    img_norm[1] = np.maximum(np.minimum(img_norm[1], 2.0), -1.0)
    img_norm = np.squeeze(np.swapaxes(np.asarray(img_norm), 0, -1))
    # pdb.set_trace()
    ## Transfer from HSD to RGB color coordinates
    X_conv = HSD2RGB_Numpy(img_norm[np.newaxis,...])
    X_conv = np.minimum(X_conv,1.0)
    X_conv = np.maximum(X_conv,0.0)
    X_conv *= 255
    X_conv = X_conv.astype(np.uint8)
    return np.squeeze(X_conv)






