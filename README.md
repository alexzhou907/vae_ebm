# Joint Training of Variational Auto-Encoder and Latent Energy-Based Model

Implementation for [Joint Training of Variational Auto-Encoder and Latent Energy-Based Model](https://arxiv.org/abs/2006.06059).

Train on Cifar10
python train_cifar.py 

Note: the codes was tested on Unbuntu with GPU V100 and rtx 2070 super. Other platforms may/may not have numerical instablities. The fid/mse can be relatively stable while the latent EBM is not (as stated in the paper), we tend to develop the robust ebm learning in the future work. 

Pretrained model can be found [here](https://www.dropbox.com/s/a3xydf594fzaokl/cifar10_pretrained.rar?dl=0)