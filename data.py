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


class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=32, protocol=None):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            n_splits = len(path_imgs) // 1000
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)