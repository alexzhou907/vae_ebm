# author: enijkamp@ucla.edu

import os
import pickle
import numpy as np
import torch.utils.data
import PIL

from torchvision import datasets

from torch.utils import data

import PIL

class UniformDataset(data.Dataset):
    def __init__(self, imageSize, nc, len):
        self.imageSize = imageSize
        self.nc = nc
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        X = torch.zeros(self.nc, self.imageSize, self.imageSize).uniform_(-1, 1)

        return X

class ConstantDataset(data.Dataset):
    def __init__(self, imageSize, nc, len):
        self.imageSize = imageSize
        self.nc = nc
        self.len = len


    def __len__(self):
        return self.len

    def __getitem__(self, i):
        n = torch.FloatTensor(1).uniform_(-1,1)
        X = n * torch.ones(self.nc,self.imageSize, self.imageSize)

        return X


class DTDDataset(datasets.ImageFolder):
    def __init__(self, imageSize, *args, **kwargs):
        super(DTDDataset, self).__init__(*args, **kwargs)
        self.imageSize = imageSize
    def __getitem__(self, index):
        data = PIL.Image.open(self.imgs[index][0])
        transoform_data = self.transform(data)[:,:self.imageSize,:self.imageSize]
        return transoform_data + torch.FloatTensor(transoform_data.shape).uniform_(-1/512, 1/512)  ## to be consistent with openai code


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)