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
import torchvision.utils as vutils
import torch.utils.data

import numpy as np
import argparse

import fid_v2_tf as fid_v2
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from data import IgnoreLabelDataset, UniformDataset, DTDDataset
from reconstruction_metric import mse_score


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

    parser.add_argument('--nez', type=int, default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', type=int, default=48)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nif', type=int, default=48)

    parser.add_argument('--energy_form', default='softplus', help='tanh | sigmoid | identity | softplus')

    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--lrE', type=float, default=0.0001, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0003, help='learning rate for GI, default=0.0002')
    parser.add_argument('--lrI', type=float, default=0.0003, help='learning rate for GI, default=0.0002')

    parser.add_argument('--beta1E', type=float, default=0., help='beta1 for adam E')
    parser.add_argument('--beta1G', type=float, default=0., help='beta1 for adam G')
    parser.add_argument('--beta1I', type=float, default=0., help='beta1 for adam I')

    parser.add_argument('--lamb', type=float, default=1.0, help='factor for reconstruction in ALI')

    parser.add_argument('--Gsteps', type=int, default=1, help='number of GI steps for each G step')
    parser.add_argument('--Isteps', type=int, default=1, help='number of GI steps for each I step')
    parser.add_argument('--Esteps', type=int, default=1, help='number of E steps for each E step')

    parser.add_argument('--is_grad_clampE', type=bool, default=False, help='whether doing the gradient clamp for E')
    parser.add_argument('--max_normE', type=float, default=100, help='max norm allowed for E')

    parser.add_argument('--is_grad_clampG', type=bool, default=False, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')

    parser.add_argument('--is_grad_clampI', type=bool, default=False, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')


    parser.add_argument('--e_decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--i_decay', type=float, default=1e-4, help='weight decay for I')
    parser.add_argument('--g_decay', type=float, default=1e-4, help='weight decay for G')

    parser.add_argument('--e_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--i_gamma', type=float, default=0.998, help='lr decay for I')
    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for G')

    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")

    parser.add_argument('--visIter', default=1, help='visualization freq')
    parser.add_argument('--evalIter', default=50, help='eval freq')
    parser.add_argument('--saveIter', default=50, help='save freq')
    parser.add_argument('--diagIter', default=1, help='diagnosis freq')
    parser.add_argument('--print_freq', type=int, default=0, help='print frequency')

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


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
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
    dataset = IgnoreLabelDataset(datasets.CIFAR10(root=os.path.join(opt.dataroot, 'cifar10'), download=True,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ])))

    test_dataset = IgnoreLabelDataset(datasets.CIFAR10(root=os.path.join(opt.dataroot, 'cifar10'), download=True, train=False,
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

    dataset = IgnoreLabelDataset(datasets.CIFAR10(root=os.path.join(opt.dataroot, 'cifar10'), train=False,
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
        dataset = IgnoreLabelDataset(datasets.SVHN(root=os.path.join(opt.dataroot, 'svhn'), download=True, split='test',
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])))
    elif opt.target_dataset == 'cifar10_train':

        dataset = IgnoreLabelDataset(datasets.CIFAR10(root=os.path.join(opt.dataroot, 'cifar'), train=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ])))
    elif opt.target_dataset == 'random':


        dataset = UniformDataset(opt.imageSize, 3, length)

    elif opt.target_dataset == 'texture':
        dataset = DTDDataset(os.path.join(opt.dataroot, 'dtd/images'),transform=transforms.Compose([
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
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
"""
netG
"""
class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
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
        self.conv52 = nn.Conv2d(nif*8, nz, 4, 1, 0)

    def forward(self, input):
        out = self.main(input)
        oI_mu = self.conv51(out)
        oI_log_sigma = self.conv52(out)

        return oI_mu, oI_log_sigma

"""
netE
"""

from torch.nn.utils import spectral_norm

class _netE(nn.Module):
    def __init__(self, nc, nz, nez, ndf):

        super(_netE, self).__init__()

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
        if opt.energy_form == 'tanh':
            energy = F.tanh(-disc_score.squeeze())
        elif opt.energy_form == 'sigmoid':
            energy = F.sigmoid(disc_score.squeeze())
        elif opt.energy_form == 'identity':
            energy = disc_score.squeeze()
        elif opt.energy_form == 'softplus':
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
            std = torch.exp(log_sigma.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def getGradNorm(net):
        pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
        gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
        return pNorm, gradNorm


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
        auroc_scores = []
        for target_dataset in ['svhn', 'random', 'texture', 'celeba']:
            opt.target_dataset = target_dataset
            ood_dataloader, ood_dataset = get_ood_dataset(opt, cifar_dataset)

            ## AUROC

            ## cifar
            cifar_labels = np.ones(len(cifar_dataset))
            cifar_scores = []
            for i, data in enumerate(cifar_dataloader, 0):
                if opt.cuda:
                    data = data.cuda()

                disc_score_T = netE(data)
                Eng_T = compute_energy(disc_score_T).detach().cpu().numpy()

                cifar_scores.append(- Eng_T)

            cifar_scores = np.concatenate(cifar_scores)

            ## ood dataset
            ood_labels = np.zeros(len(ood_dataset))
            ood_scores = []
            for i, data in enumerate(ood_dataloader, 0):
                if opt.cuda:
                    data = data.cuda()

                disc_score_T = netE(data)
                Eng_T = compute_energy(disc_score_T).detach().cpu().numpy()

                ood_scores.append(- Eng_T)

            ood_scores = np.concatenate(ood_scores)

            y_true = np.concatenate([cifar_labels, ood_labels])
            y_scores = np.concatenate([cifar_scores, ood_scores])
            auroc = roc_auc_score(y_true, y_scores)

            auroc_scores.append(auroc)

        train_flag()
        return auroc_scores

    '''
    setup auxiliaries
    '''
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    outf_recon = output_subdirs + '/recon'
    outf_syn = output_subdirs + '/syn'
    try:
        os.makedirs(outf_recon)
        os.makedirs(outf_syn)
    except OSError:
        pass

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
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, weight_decay=opt.g_decay, betas=(opt.beta1G, 0.9))
    if opt.netG != '':
        ckpt = torch.load(opt.netG)
        netG.load_state_dict(ckpt['model_state'])
        optimizerG.load_state_dict(ckpt['optimizer_state'])
    print(netG)

    netE = _netE(nc, nz, nez, ndf)
    netE.apply(weights_init)
    if opt.cuda:
        netE.cuda()
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lrE, weight_decay=opt.e_decay, betas=(opt.beta1E, 0.9))
    if opt.netE != '':
        ckpt = torch.load(opt.netE)
        netE.load_state_dict(ckpt['model_state'])
        optimizerE.load_state_dict(ckpt['optimizer_state'])

    print(netE)

    netI = _netI(nc, nz, nif)
    netI.apply(weights_init)
    if opt.cuda:
        netI.cuda()
    optimizerI = optim.Adam(netI.parameters(), lr=opt.lrI, weight_decay=opt.i_decay, betas=(opt.beta1I, 0.9))
    if opt.netI != '':
        ckpt = torch.load(opt.netI)
        netI.load_state_dict(ckpt['model_state'])
        optimizerI.load_state_dict(ckpt['optimizer_state'])

    print(netI)

    if opt.netG != '' and  opt.netE != '' and  opt.netI != '':
        start_epoch = torch.load(opt.netI)['epoch']
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


    for epoch in range(start_epoch, opt.niter):

        lrE_schedule.step(epoch=epoch)
        lrG_schedule.step(epoch=epoch)
        lrI_schedule.step(epoch=epoch)


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


                errE.backward()
                if i % opt.print_freq == 0:
                    netEpNorm, netEgradNorm = getGradNorm(netE)

                    norm_z_sigma = z_input_log_sigma.exp().mean()
                    norm_z_T = z_input.abs().mean()
                    norm_z_F = noise.abs().mean()

                if opt.is_grad_clampE:
                    torch.nn.utils.clip_grad_norm_(netE.parameters(), opt.max_normE)

                optimizerE.step()

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


                """
                compute the q(z|x) entropy: which is -log_q_z
                """

                errI = opt.lamb * (errRecon + errKld) - (E_T + (-log_q_z))
                errI.backward()

                if i % opt.print_freq == 0:
                    netIpNorm, netIgradNorm = getGradNorm(netI)
                    entropyi = torch.mean(torch.sum(z_input_log_sigma.squeeze(), dim=1))

                if opt.is_grad_clampI:
                    torch.nn.utils.clip_grad_norm_(netI.parameters(), opt.max_normI)
                optimizerI.step()

            for gisteps in range(opt.Gsteps):
                netG.zero_grad()

                x_gen = netG(noiseV)

                disc_score_F = netE(x_gen, noiseV)
                Eng_F = compute_energy(disc_score_F)
                E_F = torch.mean(Eng_F)

                z_input_mu, z_input_log_sigma = netI(inputV)
                z_input = reparametrize(z_input_mu, z_input_log_sigma)

                inputV_recon = netG(z_input.detach())
                errRecon = mse_loss(inputV_recon, inputV) / batch_size

                errG = opt.lamb * errRecon + E_F
                errG.backward()

                if i % opt.print_freq == 0:
                    netGpNorm, netGgradNorm = getGradNorm(netG)

                if opt.is_grad_clampG:
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.max_normG)
                optimizerG.step()

            if i % opt.print_freq == 0:
                logging.info('[%3d/%3d][%3d/%3d] errG: %10.2f, errI: %10.2f, errE: %10.2f, '
                             'z_T: %10.2f, z_F: %10.2f, z_sigma: %10.2f, E_T: %10.2f, E_F: %10.2f, log_q_z: %10.2f, log_p_z: %10.2f, '
                             'errRecon: %10.2f, entropyi(sigma): %10.2f, net G param:%10.2f, grad param norm: %10.2f. '
                             'net I param:%10.2f, grad param norm: %10.2f, net E param:%10.2f, grad param norm: %10.2f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errG.data.item(), errI.data.item(), errE.data.item(),
                         norm_z_T.data.item(), norm_z_F.data.item(), norm_z_sigma.data.item(),
                         E_T.data.item(), E_F.data.item(),
                         log_q_z.data.item(), log_p_z.data.item(), errRecon.data.item(),
                         entropyi.data.item(),
                         netGpNorm.data.item(), netGgradNorm.data.item(),
                         netIpNorm.data.item(), netIgradNorm.data.item(),
                         netEpNorm.data.item(), netEgradNorm.data.item(),
                         ))


        if (epoch+1)%opt.visIter==0:
            with torch.no_grad():

                gen_samples = netG(fixed_noiseV)
                vutils.save_image(gen_samples.data, '%s/epoch_%03d_iter_%03d_samples.png' % (outf_syn, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                infer_z_mu_input, _ = netI(inputV)
                recon_input = netG(infer_z_mu_input)
                vutils.save_image(recon_input.data, '%s/epoch_%03d_iter_%03d_recon_input.png' % (outf_recon, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                infer_z_mu_sample, _ = netI(gen_samples)
                recon_sample = netG(infer_z_mu_sample)
                vutils.save_image(recon_sample.data, '%s/epoch_%03d_iter_%03d_recon_samples.png' %(outf_syn, epoch, i), normalize=True, nrow=int(np.sqrt(opt.batchSize)))

                """
                visualize the interpolation
                z1 = I(x1), z2 = I(x2), z1---> z2, z = z1 + alpha*(z2-z1), then x = G(z)
                """
                infer_z_mu_input, _ = netI(inputV)
                between_input_list = [inputV.data.cpu().numpy()]  # NUMPY list
                zfrom = infer_z_mu_input.data.cpu()
                perm = torch.arange(1, zfrom.shape[0] + 1)
                perm[-1] = 0

                zto = infer_z_mu_input[perm.long()].data.cpu()
                fromto = zto - zfrom
                for alpha in np.linspace(0, 1, 8):
                    between_z = zfrom + alpha * fromto
                    recon_between = netG(Variable(between_z.cuda()))
                    between_input_list.append(recon_between.data.cpu().numpy())
                between_input_list.append(inputV[perm.long()].data.cpu().numpy())
                between_canvas_np = np.stack(between_input_list, axis=1).reshape(-1, 3, opt.imageSize, opt.imageSize)
                vutils.save_image(torch.from_numpy(between_canvas_np), '%s/epoch_%03d_iter_%03d_interpolate_train.png' % (outf_syn, epoch, i), normalize=True, nrow=10, padding=5)


        if (epoch+1) % opt.saveIter == 0:

            opt_dict = {'netG': (netG, optimizerG), 'netE': (netE, optimizerE), 'netI': (netI, optimizerI)}

            for key in opt_dict:
                save_dict = {
                    'epoch': epoch+1,
                    'model_state': opt_dict[key][0].state_dict(),
                    'optimizer_state': opt_dict[key][1].state_dict()
                }
                torch.save(save_dict, '%s/%s_epoch_%d.pth' % (output_dir, key, (epoch+1)))
        if (epoch+1) % opt.evalIter == 0:

            train_flag()

            # get the fid v2 score
            to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
            with torch.no_grad():
                gen_samples = torch.cat([netG(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
                gen_samples_np = 255 * unnormalize(gen_samples.numpy())
                gen_samples_np = to_nhwc(gen_samples_np)
                fid = fid_v2.fid_score(create_session, 255 * to_nhwc(unnormalize(dataset_full)), gen_samples_np)

            mse_val = mse_score(test_dataloader, netG, netI, opt.imageSize, 100, output_subdirs + '/mse_test/',
                                output_subdirs + '/mse_recon/', epoch)
            auroc = get_auroc()

            logging.info("FID:{}, MSE:{}, AUROC:[{}, {}, {}, {}]".format(fid, mse_val, *auroc))




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
