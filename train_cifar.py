# 49 -> Inception:5.248693466186523, Inception std:0.04255198314785957, MSE:91.596125

import os
import random

from shutil import copyfile

import datetime
import logging
import sys

import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.utils.data

from adam_lr import Adam

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

import fid_v2_tf as fid_v2
from inception_score_v2_tf import get_inception_score

from reconstruction_metric import mse_score

from sklearn.metrics import roc_auc_score
from data import IgnoreLabelDataset, ConstantDataset, UniformDataset, DTDDataset

from torch.utils import data
def parse_args():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataset', default='cifar10',  help='SVHN|cifar10 | lsun | imagenet | folder | lfw | fake')
    #parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataroot', default='./data/cifar/', help='path to dataset')
    parser.add_argument('--dataroot_test', default='./training_images/CelebA1000', help='path to test dataset')

    parser.add_argument('--target_dataset', default='random',
                        help=' svhn | random | constant | texture | cifar10_train')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--sampleSize', type=int, default=100, help='sample size used for generation evaluation')
    parser.add_argument('--sampleRun', type=int, default=10, help='the number of times we compute inception like score')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

    parser.add_argument('--nez', type=int, default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', type=int, default=48)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nif', type=int, default=48)

    parser.add_argument('--energy_form', default='softplus', help='tanh | sigmoid | identity | softplus')

    parser.add_argument('--niter', type=int, default=800, help='number of epochs to train for')
    parser.add_argument('--lrE', type=float, default=0.0001, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0003, help='learning rate for GI, default=0.0002')
    parser.add_argument('--lrI', type=float, default=0.0003, help='learning rate for GI, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0., help='beta1 for adam. default=0.5')
    parser.add_argument('--beta1G', type=float, default=0., help='beta1 for adam GI. default=0.5')
    parser.add_argument('--beta1I', type=float, default=0., help='beta1 for adam GI. default=0.5')

    parser.add_argument('--lamb', type=float, default=1.0, help='factor for reconstruction in ALI')
    parser.add_argument('--recon', type=float, default=1.0, help='factor for reconstruction in ALI')

    parser.add_argument('--Gsteps', type=int, default=1, help='number of GI steps for each G step')
    parser.add_argument('--Isteps', type=int, default=1, help='number of GI steps for each I step')
    parser.add_argument('--Esteps', type=int, default=1, help='number of E steps for each E step')

    parser.add_argument('--is_grad_clampE', type=bool, default=False, help='whether doing the gradient clamp for E')
    parser.add_argument('--max_normE', type=float, default=100, help='max norm allowed for E')

    parser.add_argument('--is_grad_clampG', type=bool, default=False, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')

    parser.add_argument('--is_grad_clampI', type=bool, default=False, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')
    parser.add_argument('--spectral_norm', type=bool, default=False, help='spectral norm for EBM')

    parser.add_argument('--e_decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--i_decay', type=float, default=1e-4, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=1e-4, help='weight decay for G')

    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for G')

    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")

    parser.add_argument('--visIter', default=1, help='number of iterations we need to visualize')
    parser.add_argument('--plotIter', default=1, help='number of iterations we need to visualize')
    parser.add_argument('--evalIter', default=50, help='number of iterations we need to evaluate')
    parser.add_argument('--saveIter', default=50, help='number of epochs we need to save the model')
    parser.add_argument('--diagIter', default=1, help='number of epochs we need to save the model')

    parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', default=42, type=int, help='42 is the answer to everything')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        opt.cuda = True

    return opt

def set_global_gpu_env(opt):
    torch.cuda.set_device(opt.gpu)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def get_output_dir(exp_id, fs_prefix='../../../data5/tian/vae_ebm/'):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix + 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def unnormalize(img):
    return img / 2.0 + 0.5



def set_seed(opt):

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False


def get_dataset(opt):
    from data import IgnoreLabelDataset
    dataset = IgnoreLabelDataset(datasets.CIFAR10(root=opt.dataroot, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ])))

    test_dataset = IgnoreLabelDataset(datasets.CIFAR10(root=opt.dataroot, download=True, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    dataset_full = np.array([x.cpu().numpy() for x in iter(dataset)])

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))

    return dataloader, dataset_full, test_dataloader

################################### AUROC ##############################
def get_cifar_dataset(opt):

    dataset = IgnoreLabelDataset(datasets.CIFAR10(root='data/cifar/', train=False,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ])))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    return dataloader, dataset

def get_ood_dataset(opt, cifar_dataset):
    length = len(cifar_dataset)

    if opt.target_dataset == 'svhn':
        dataset = IgnoreLabelDataset(datasets.SVHN(root='data/svhn/', download=True, split='test',
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])))
    elif opt.target_dataset == 'cifar10_train':

        dataset = IgnoreLabelDataset(datasets.CIFAR10(root='data/cifar/', train=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ])))
    elif opt.target_dataset == 'random':


        dataset = UniformDataset(opt.imageSize, 3, length)
    elif opt.target_dataset == 'constant':


        dataset = ConstantDataset(opt.imageSize, 3, length)

    elif opt.target_dataset == 'texture':
        dataset = DTDDataset('data/dtd/images/',transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.CenterCrop(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ]))
    else:
        raise ValueError('no dataset')


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    return dataloader, dataset


##################################


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_()
        # nn.init.xavier_normal_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
        #m.weight.data.fill_(0.001)
    #elif classname.find('Linear') != -1:
        #xavier_uniform(m.weight)
    elif classname.find('BatchNorm') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
"""
netG
"""
class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        # nz is the input channel
        # nc is the output channel
        # ngf is the hiddle channel 32
        # for CIFAR10
        super(_netG, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf*2 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf *2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, nc, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, input):

        oG_out = self.main(input)
        return oG_out
"""
netI
"""
class _netI(nn.Module):
    def __init__(self, nc, nz, nif):
        super(_netI, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nif, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nif),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif, nif*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(nif*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif *2, nif * 4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(nif * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nif * 4, nif * 8, 4, 2, 1,bias=False),
            nn.BatchNorm2d(nif * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv51 = nn.Conv2d(nif*8, nz, 4, 1, 0)
        self.conv52 = nn.Conv2d(nif*8, nz, 4, 1, 0) # for log_sigma

    def forward(self, input):
        out = self.main(input)
        oI_mu = self.conv51(out)
        oI_log_sigma = self.conv52(out) # actually its log variance
        # [batch nz 1 1]
        return oI_mu, oI_log_sigma

"""
netE
"""

from torch.nn.utils import spectral_norm

class _netE(nn.Module):
    """
    discriminator is based on x and z jointly
    x is first go through conv-layers
    z is first go through conv-layers
    then (x, z) is concatenate along the axis (both ndf*4), then go through several layers
    """
    def __init__(self, nc, nz, nez, ndf):
        # FOR CIFAR10
        super(_netE, self).__init__()

        # spectral_norm = lambda x: x

        self.x = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*8, ndf*8, 4, 1, 0)),
        )

        ## z

        self.z = nn.Sequential(

            spectral_norm(nn.Conv2d(nz, ndf, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf , 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf * 2, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf *4, ndf * 4, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*8, ndf*8, 1, 1, 0)),
        )

        self.xz = nn.Sequential(

            spectral_norm(nn.Conv2d(ndf*16 , ndf*16, 1, 1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ndf*16, nez, 1, 1, 0)),

        )

    def forward(self, input, z, leak=0.1):

        ox = self.x(input)
        oz = self.z(z)
        oxz = torch.cat([ox, oz], 1)
        oE_outxz = self.xz(oxz)
        return oE_outxz


def train(opt, output_dir):

    '''
    first define necessary functions
    '''

    # define energy and diagonal Normal log-probability
    def compute_energy(disc_score):
        # disc score: batch nez 1 1
        # return shape: [batch]
        if opt.energy_form == 'tanh':
            energy = F.tanh(-disc_score.squeeze())
        elif opt.energy_form == 'sigmoid':
            energy = F.sigmoid(disc_score.squeeze())
        elif opt.energy_form == 'identity':
            energy = disc_score.squeeze()
        elif opt.energy_form == 'softplus':
            # energy = F.softplus(-torch.sum(disc_score.squeeze(), dim=1))
            energy = F.softplus(disc_score.squeeze())
        return energy

    def diag_normal_NLL(z, z_mu, z_log_sigma):
        # z_log_sigma: log variance
        # sum over the dimension, but leave the batch dim unchanged
        # define the Negative Log Probablity of Normal which has diagonal cov
        # input:[batch nz, 1, 1] squeeze it to batch nz
        # return shape is [batch]
        nll = 0.5 * torch.sum(z_log_sigma.squeeze(), dim=1) + \
              0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
        return nll.squeeze()

    def diag_standard_normal_NLL(z):
        # get the negative log-likelihood of standard normal distribution
        nll = 0.5 * torch.sum((torch.mul(z, z)).squeeze(), dim=1)
        return nll.squeeze()

    def reparametrize(mu, log_sigma, is_train=True):
        if is_train:
            # std = log_sigma.exp_() # This line has potential bugs cause it changes the log_sigma
            std = torch.exp(log_sigma.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def getGradNorm(net):
        pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
        gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
        return pNorm, gradNorm

    def plot_stats():
        p_i = 1
        p_n = len(stat_1)

        f = plt.figure(figsize=(20, p_n * 5))

        def plot(stats, stats_i):
            nonlocal p_i
            for j, (k, v) in enumerate(stats.items()):
                plt.subplot(p_n, 1, p_i)
                plt.plot(stats_i, v)
                plt.ylabel(k)
                p_i += 1

        plot(stat_1, stat_1_i)

        f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
        plt.close(f)

    def eval_flag():
        netG.eval()
        netI.eval()
        netE.eval()

    def train_flag():
        netG.train()
        netI.train()
        netE.train()

    # init tf
    def create_session():
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.visible_device_list = str(opt.gpu)
        return tf.Session(config=config)

    def get_auroc():
        eval_flag()

        cifar_dataloader, cifar_dataset = get_cifar_dataset(opt)
        ood_dataloader, ood_dataset = get_ood_dataset(opt, cifar_dataset)

        ## AUROC

        ## cifar
        cifar_labels = np.ones(len(cifar_dataset))
        cifar_scores = []
        for i, data in enumerate(cifar_dataloader, 0):
            if opt.cuda:
                data = data.cuda()

            z_input_mu, _ = netI(data)

            disc_score_T = netE(data, z_input_mu.detach())
            Eng_T = compute_energy(disc_score_T).detach().cpu().numpy()

            cifar_scores.append(- Eng_T)

        cifar_scores = np.concatenate(cifar_scores)

        ## ood dataset
        ood_labels = np.zeros(len(ood_dataset))
        ood_scores = []
        for i, data in enumerate(ood_dataloader, 0):
            if opt.cuda:
                data = data.cuda()

            z_input_mu, _ = netI(data)

            disc_score_T = netE(data, z_input_mu.detach())
            Eng_T = compute_energy(disc_score_T).detach().cpu().numpy()

            ood_scores.append(- Eng_T)

        ood_scores = np.concatenate(ood_scores)

        y_true = np.concatenate([cifar_labels, ood_labels])
        y_scores = np.concatenate([cifar_scores, ood_scores])
        auroc = roc_auc_score(y_true, y_scores)

        train_flag()
        return auroc

    '''
    setup auxiliaries
    '''
    output_subdirs = output_dir + opt.outf
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    outf_recon = output_subdirs + '/recon'
    outf_syn = output_subdirs + '/syn'
    outf_err = output_subdirs + '/curve'
    try:
        os.makedirs(outf_recon)
        os.makedirs(outf_syn)
        os.makedirs(outf_err)
    except OSError:
        pass
    ## open file for later use
    out_f = open("%s/results.txt" % output_dir, 'w')

    ## get constants
    nz = int(opt.nz)
    nez = int(opt.nez)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nif = int(opt.nif)
    nc = 3

    dataloader, dataset_full, test_dataloader = get_dataset(opt)

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_()  # for visualize

    mse_loss = nn.MSELoss(size_average=False)

    new_noise = lambda: noise.resize_(opt.batchSize, nz, 1, 1).normal_()
    num_samples = 50000

    '''
    create networks
    '''

    netG = _netG(nz, nc, ngf)
    netG.apply(weights_init)
    if opt.cuda:
        netG.cuda()
    optimizerG = Adam(netG.parameters(), lr=opt.lrG, weight_decay=opt.g_decay, betas=(opt.beta1G, 0.9))
    if opt.netG != '':
        ckpt = torch.load(opt.netG)
        netG.load_state_dict(ckpt['model_state'])
        optimizerG.load_state_dict(ckpt['optimizer_state'])
    print(netG)

    netE = _netE(nc, nz, nez, ndf)
    netE.apply(weights_init)
    if opt.cuda:
        netE.cuda()
    optimizerE = Adam(netE.parameters(), lr=opt.lrE, weight_decay=opt.e_decay, betas=(opt.beta1, 0.9))
    if opt.netE != '':
        ckpt = torch.load(opt.netE)
        netE.load_state_dict(ckpt['model_state'])
        optimizerE.load_state_dict(ckpt['optimizer_state'])

    print(netE)

    netI = _netI(nc, nz, nif)
    netI.apply(weights_init)
    if opt.cuda:
        netI.cuda()
    optimizerI = Adam(netI.parameters(), lr=opt.lrI, weight_decay=opt.i_decay, betas=(opt.beta1I, 0.9))
    if opt.netI != '':
        ckpt = torch.load(opt.netI)
        netI.load_state_dict(ckpt['model_state'])
        optimizerI.load_state_dict(ckpt['optimizer_state'])

    print(netI)

    if opt.netG != '' and  opt.netE != '' and  opt.netI != '':
        start_epoch = torch.load(opt.netI)['epoch'] + 1
    else:
        start_epoch = 0

    if opt.cuda:
        input = input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        mse_loss.cuda()

    fixed_noiseV = Variable(fixed_noise)

    lrE_schedule = optim.lr_scheduler.ExponentialLR(optimizerE, opt.e_gamma)
    lrG_schedule = optim.lr_scheduler.ExponentialLR(optimizerG, opt.g_gamma)
    lrI_schedule = optim.lr_scheduler.ExponentialLR(optimizerI, opt.i_gamma)

    '''
    define stats
    '''
    # TODO merge with code below, print to file and plot pdf
    stats_headings = [['epoch',          '{:>14}',  '{:>14d}'],
                      ['err(I)',         '{:>14}',  '{:>14.3f}'],
                      ['err(G)',         '{:>14}',  '{:>14.3f}'],
                      ['err(E)',         '{:>14}',  '{:>14.3f}'],
                      ['norm(grad(E))',  '{:>14}',  '{:>14.3f}'],
                      ['norm(grad(I))',  '{:>14}',  '{:>14.3f}'],
                      ['norm(grad(G))',  '{:>14}',  '{:>14.3f}'],
                      ['entropy',         '{:>14}',  '{:>14.3f}'],
                      ['inception',            '{:>14}',  '{:>14.3f}'],
                      ['inception_std', '{:>14}', '{:>14.3f}'],
                      ['fid', '{:>14}', '{:>14.3f}'],
                      ['auroc', '{:>14}', '{:>14.3f}'],
                      ['mse(val)',       '{:>14}',  '{:>14.3f}']]

    stat_1_i = []
    stat_1 = {
        'auroc': [],

        'fid': [],
        'mse_score': [],
        'inception': [],
        'inception_std':[],
        'err(I)': [],
        'err(G)': [],
        'err(E)': [],
        'norm(grad(E))': [],
        'norm(grad(I))': [],
        'norm(grad(G))': [],
        'norm(weight(E))': [],
        'norm(weight(I))': [],
        'norm(weight(G))': [],
        'entropy': [],
        'E_T': [],
        'E_F': [],
        'E_T-E_F': [],
        'errRecon': [],
        'log_q_z': [],
        'log_p_z': [],
        'norm(z_T)':[],
        'norm(z_F)':[],
        'norm(z_sigma)':[],
    }
    for k in range(len(list(netG.parameters()))):
        stat_1['lrG{}'.format(k)] = []
    for k in range(len(list(netI.parameters()))):
        stat_1['lrI{}'.format(k)] = []
    for k in range(len(list(netE.parameters()))):
        stat_1['lrE{}'.format(k)] = []

    fid = 0.0
    inception = 0.0
    inception_std = 0.0
    mse_val = 0.0
    auroc = 0

    # get the fid v2 score
    # to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
    # with torch.no_grad():
    #     gen_samples = torch.cat([netG(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
    #     gen_samples_np = 255 * unnormalize(gen_samples.numpy())
    #     gen_samples_np = to_nhwc(gen_samples_np)
    #     gen_samples_list = [gen_samples_np[i, :, :, :] for i in range(gen_samples_np.shape[0])]
    #     inception, inception_std = get_inception_score(create_session, gen_samples_list)
    #
    #
    # logging.info("FID:{}, IS:{}, MSE:{}, AUROC:{}".format(fid, inception, mse_val, auroc))


    for epoch in range(start_epoch, opt.niter):

        lrE_schedule.step(epoch=epoch)
        lrG_schedule.step(epoch=epoch)
        lrI_schedule.step(epoch=epoch)

        stats_values = {k[0]: 0 for k in stats_headings}
        stats_values['epoch'] = epoch

        num_batch = len(dataloader.dataset) / opt.batchSize
        for i, data in enumerate(dataloader, 0):
            """
            Train joint EBM
            """
            real_cpu = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputV = Variable(input)

            for esteps in range(opt.Esteps):

                netE.zero_grad()
                """
                get the pair (p_d, q(z|x)) i.e., training and inference
                """
                z_input_mu, z_input_log_sigma = netI(inputV)
                z_input = reparametrize(z_input_mu, z_input_log_sigma)
                disc_score_T = netE(inputV, z_input.detach())
                Eng_T = compute_energy(disc_score_T)
                E_T = torch.mean(Eng_T)


                """
                get the pair {p(z), p(x|z)} i.e., prior z and the generated x
                """
                noise.resize_(batch_size, nz, 1, 1).normal_() # or Uniform
                noiseV = Variable(noise)
                x_gen = netG(noiseV)
                disc_score_F = netE(x_gen.detach(), noiseV)
                Eng_F = compute_energy(disc_score_F)
                E_F = torch.mean(Eng_F)
                errE = E_T - E_F

                norm_z_sigma = z_input_log_sigma.exp().mean()
                norm_z_T = z_input.abs().mean()
                norm_z_F = noise.abs().mean()

                errE.backward()
                netEpNorm, netEgradNorm = getGradNorm(netE)
                if opt.is_grad_clampE:
                    torch.nn.utils.clip_grad_norm_(netE.parameters(), opt.max_normE)

                _, lrE = optimizerE.step()

            """
             Train I
             (1) trying to get close to the joint energy based model
             (2) trying to larger the entropy of posterior distribution
             (3) trying to reconstruct the training data
             """
            for esteps in range(opt.Isteps):
                netI.zero_grad()

                disc_score_T = netE(inputV, z_input)
                Eng_T = compute_energy(disc_score_T)
                E_T = torch.mean(Eng_T)

                inputV_recon = netG(z_input)
                errRecon = mse_loss(inputV_recon, inputV) / batch_size
                """
                for kl term, we dont use the analytic form, but use their expectation form
                """
                log_p_z = torch.mean(diag_standard_normal_NLL(z_input))  # [batch]--> scalar
                log_q_z = torch.mean(diag_normal_NLL(z_input, z_input_mu, z_input_log_sigma))  # [batch]---> scalar
                # errKld: KL[q(z|x)|p(z)]
                errKld = -log_q_z + log_p_z

                entropyi = torch.mean(torch.sum(z_input_log_sigma.squeeze(), dim=1))
                """
                compute the q(z|x) entropy: which is -log_q_z
                """

                # errI1 = opt.lamb * (errRecon + errKld)
                # errI1.backward(retain_graph=True)
                # _, netIgradNorm1 = getGradNorm(netI)
                

                # errI2 = E_T
                # errI2.backward(retain_graph=True)
                # _, netIgradNorm2 = getGradNorm(netI)
                

                #z_fake_mu, z_fake_log_sigma = netI(x_gen.detach())
                #errLatent = torch.mean(diag_normal_NLL(noiseV, z_fake_mu, z_fake_log_sigma))

                errI = opt.lamb * (opt.recon*errRecon + 1*errKld) - (E_T + (-log_q_z))
                errI.backward()
                netIpNorm, netIgradNorm = getGradNorm(netI)

                if opt.is_grad_clampI:
                    torch.nn.utils.clip_grad_norm_(netI.parameters(), opt.max_normI)
                _, lrI = optimizerI.step()

            for gisteps in range(opt.Gsteps):
                netG.zero_grad()

                x_gen = netG(noiseV)

                disc_score_F = netE(x_gen, noiseV)
                Eng_F = compute_energy(disc_score_F)
                E_F = torch.mean(Eng_F)

                z_input_mu, z_input_log_sigma = netI(inputV)
                z_input = reparametrize(z_input_mu, z_input_log_sigma)
                #
                inputV_recon = netG(z_input.detach())
                errRecon = mse_loss(inputV_recon, inputV) / batch_size

                

                errG = opt.lamb * opt.recon*errRecon + E_F
                errG.backward(retain_graph=True)

               
                netGpNorm, netGgradNorm = getGradNorm(netG)

                if opt.is_grad_clampG:
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.max_normG)
                _, lrG = optimizerG.step()

            logging.info('[%3d/%3d][%3d/%3d] errG: %10.2f, errI: %10.2f, errE: %10.2f, lrG: %10.6f, lrI: %10.6f, lrE: %10.6f, z_T: %10.2f, z_F: %10.2f, z_sigma: %10.2f, E_T: %10.2f, E_F: %10.2f, log_q_z: %10.2f, log_p_z: %10.2f, errRecon: %10.2f, entropyi(sigma): %10.2f, net G param:%10.2f, grad param norm: %10.2f, net I param:%10.2f, grad param norm: %10.2f, net E param:%10.2f, grad param norm: %10.2f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errG.data.item(), errI.data.item(), errE.data.item(),
                     np.mean(lrG), np.mean(lrI), np.mean(lrE),
                     norm_z_T.data.item(), norm_z_F.data.item(), norm_z_sigma.data.item(),
                     E_T.data.item(), E_F.data.item(),
                     log_q_z.data.item(), log_p_z.data.item(), errRecon.data.item(),
                     entropyi.data.item(),
                     netGpNorm.data.item(), netGgradNorm.data.item(),
                     netIpNorm.data.item(), netIgradNorm.data.item(), 
                     netEpNorm.data.item(), netEgradNorm.data.item(),
                     ))

            stats_values['err(G)'] += errG.data.item() / num_batch
            stats_values['err(I)'] += errI.data.item() / num_batch
            stats_values['err(E)'] += errE.data.item() / num_batch
            stats_values['norm(grad(E))'] += netEgradNorm.data.item() / num_batch
            stats_values['norm(grad(I))'] += netIgradNorm.data.item() / num_batch
            stats_values['norm(grad(G))'] += netGgradNorm.data.item() / num_batch
            stats_values['entropy'] += log_q_z.data.item() / num_batch

        # diagnostics
        if (epoch+1) % opt.diagIter == 0:
            stat_1_i.append(epoch)
            stat_1['err(I)'].append(errI.data.item())
            stat_1['err(G)'].append(errG.data.item())
            stat_1['err(E)'].append(errE.data.item())
            stat_1['norm(grad(I))'].append(netIgradNorm.data.item())
            stat_1['norm(grad(G))'].append(netGgradNorm.data.item())
            stat_1['norm(grad(E))'].append(netEgradNorm.data.item())
            stat_1['norm(weight(I))'].append(netIpNorm.data.item())
            stat_1['norm(weight(G))'].append(netGpNorm.data.item())
            stat_1['norm(weight(E))'].append(netEpNorm.data.item())
            stat_1['entropy'].append(log_q_z.data.item())
            stat_1['E_F'].append(E_F.data.item())
            stat_1['E_T'].append(E_T.data.item())
            stat_1['E_T-E_F'].append(E_T.data.item()-E_F.data.item())
            stat_1['inception'].append(inception)
            stat_1['inception_std'].append(inception_std)
            stat_1['errRecon'].append(errRecon.data.item())
            stat_1['log_q_z'].append(log_q_z.data.item())
            stat_1['log_p_z'].append(log_p_z.data.item())
            stat_1['norm(z_T)'].append(norm_z_T.data.item())
            stat_1['norm(z_F)'].append(norm_z_F.data.item())
            stat_1['norm(z_sigma)'].append(norm_z_sigma.data.item())
            stat_1['mse_score'].append(mse_val)
            stat_1['fid'].append(fid)
            stat_1['auroc'].append(auroc)

            for k, w in enumerate(lrG):
                stat_1['lrG{}'.format(k)].append(w)
            for k, w in enumerate(lrI):
                stat_1['lrI{}'.format(k)].append(w)
            for k,w in enumerate(lrE):
                stat_1['lrE{}'.format(k)].append(w)

            plot_stats()

        if (epoch+1)%opt.visIter==0:
            with torch.no_grad():
            
                gen_samples = netG(fixed_noiseV)
                vutils.save_image(gen_samples.data, '%s/epoch_%03d_iter_%03d_samples_train.png' % (outf_syn, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                infer_z_mu_input, _ = netI(inputV)
                recon_input = netG(infer_z_mu_input)
                vutils.save_image(recon_input.data, '%s/epoch_%03d_iter_%03d_reconstruct_input_train.png' % (outf_recon, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                infer_z_mu_sample, _ = netI(gen_samples)
                recon_sample = netG(infer_z_mu_sample)
                vutils.save_image(recon_sample.data, '%s/epoch_%03d_iter_%03d_reconstruct_samples_train.png' %(outf_syn, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))
            disp_str=['E_T - E_F', 'E_T', 'E_F', 'ReconX', 'KLD', 'E_I', 'E_G']
            disp_val=[errE.data.item(), E_T.data.item(), E_F.data.item(), errRecon.data.item(),errKld.data.item(), errI.data.item(), errG.data.item()]
            res_str = ('[%d] ' % i) + ", ".join("%s: %.2f" %(s,v) for s, v in zip(disp_str, disp_val))
            out_f.write(res_str+"\n")
            out_f.flush()

        if (epoch+1) % opt.saveIter == 0:

            opt_dict = {'netG': (netG, optimizerG), 'netE': (netE, optimizerE), 'netI': (netI, optimizerI)}

            for key in opt_dict:
                save_dict = {
                    'epoch': epoch,
                    'model_state': opt_dict[key][0].state_dict(),
                    'optimizer_state': opt_dict[key][1].state_dict()
                }
                torch.save(save_dict, '%s/%s_epoch_%d.pth' % (output_dir, key, epoch))
        if (epoch+1) % opt.evalIter == 0:

            # get the fid v2 score
            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
            with torch.no_grad():
                gen_samples = torch.cat([netG(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
                gen_samples_np = 255 * unnormalize(gen_samples.numpy())
                gen_samples_np = to_nhwc(gen_samples_np)
                gen_samples_list = [gen_samples_np[i, :, :, :] for i in range(gen_samples_np.shape[0])]
                fid = fid_v2.fid_score(create_session, 255 * to_nhwc(unnormalize(dataset_full)), gen_samples_np)
                inception, inception_std = get_inception_score(create_session, gen_samples_list)

            mse_val = mse_score(test_dataloader, netG, netI, opt.imageSize, 100, outf_recon)
            auroc = get_auroc()

            logging.info("FID:{}, IS:{}, MSE:{}, AUROC:{}".format(fid, inception, mse_val, auroc))
            # torch.cuda.empty_cache()



        stats_values['auroc'] = auroc
        stats_values['fid'] = fid
        stats_values['inception'] = inception
        stats_values['inception_std'] = inception_std
        stats_values['mse(val)'] = mse_val
        logging.info(''.join([h[2] for h in stats_headings]).format(*[stats_values[k[0]] for k in stats_headings]))



def main():
    opt = parse_args()
    set_global_gpu_env(opt)
    set_seed(opt)

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)
    setup_logging(output_dir)

    logging.info(opt)

    train(opt, output_dir)

if __name__ == '__main__':
    main()
