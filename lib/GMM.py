import torch
from torch import distributions, nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import pdb
import horovod.torch as hvd

    
class GMM_model(nn.Module):
    def __init__(self, input_size, args, num_clusters=0, name='GMM_Statistics'):
        super(GMM_model, self).__init__()
        self.num_clusters = num_clusters
        self.args         = args
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z, x_hsd):
        
        x_hsd = x_hsd.reshape(-1,3,self.args.img_size,self.args.img_size)
        # Get softmax over cluster (z = (batch_size,args.nclusters,img_size,img_size))
        gamma = self.softmax(z)
        D, h, s = torch.split(x_hsd, [1, 1, 1], dim=1)
        WXd = torch.mul(gamma, D.repeat(1, self.num_clusters,1,1))
        WXa = torch.mul(gamma, h.repeat(1, self.num_clusters,1,1))
        WXb = torch.mul(gamma, s.repeat([1, self.num_clusters,1,1]))
        S = torch.sum(torch.sum(gamma, dim=2), dim=2)
        S = torch.add(S, 1e-07)
        S = S.view(-1, self.num_clusters)
        M_d = torch.sum(torch.sum(WXd, dim=2), dim=2) / S
        M_a = torch.sum(torch.sum(WXa, dim=2), dim=2) / S
        M_b = torch.sum(torch.sum(WXb, dim=2), dim=2) / S

        self.mu = torch.cat((M_d,M_a,M_b), 0)
        mu = torch.split(self.mu, [1]*self.num_clusters, dim=-1)
       
        

        # (-1,256,256,4)
        Norm_d = (D.reshape(-1,self.args.img_size,self.args.img_size,1) - M_d.view(-1,1,1, self.num_clusters))**2
        Norm_h = (h.reshape(-1,self.args.img_size,self.args.img_size,1) - M_a.view(-1,1,1, self.num_clusters))**2
        Norm_s = (s.reshape(-1,self.args.img_size,self.args.img_size,1) - M_b.view(-1,1,1, self.num_clusters))**2
    
        # (-1,4,256,256)
        WSd = torch.mul(gamma, Norm_d.reshape(-1,self.num_clusters,self.args.img_size,self.args.img_size))
        WSh = torch.mul(gamma, Norm_h.reshape(-1,self.num_clusters,self.args.img_size,self.args.img_size))
        WSs = torch.mul(gamma, Norm_s.reshape(-1,self.num_clusters,self.args.img_size,self.args.img_size))
    
        # (-1,4)
        S_d = torch.sqrt(torch.sum(torch.sum(WSd, dim=2), dim=2)/ S)
        S_h = torch.sqrt(torch.sum(torch.sum(WSh, dim=2), dim=2)/ S)
        S_s = torch.sqrt(torch.sum(torch.sum(WSs, dim=2), dim=2)/ S)
    
        self.std = torch.cat((S_d, S_h, S_s), 0)
        std = torch.split(self.std, [1]*self.num_clusters, dim=-1)

        dist = list()
        for k in range(self.num_clusters):
            dist.append(torch.distributions.normal.Normal(mu[k].reshape(-1,3,1,1),std[k].reshape(-1,3,1,1)))
            
    
        # pi[0].shape = (-1,256,256,1)
        pi = torch.split(gamma, [1]*self.num_clusters, dim=1)
        prob = list()
                
        for k in range(self.num_clusters):
            prob.append(dist[k].log_prob(x_hsd))
        
        prob = [torch.exp(torch.mean(prob[k],axis=1))* torch.squeeze(pi[k]) for k in range(self.num_clusters)]
        
        prob = torch.stack(prob)
        prob = torch.min(torch.sum(prob, dim=0) + 1e-7,torch.tensor(1.0, dtype=torch.float32).cuda())
        log_prob = torch.log(prob).sum()
        return log_prob, (self.mu, self.std, gamma.permute(0,2,3,1))
