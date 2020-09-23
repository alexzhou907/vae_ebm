# Joint Training of Variational Auto-Encoder and Latent Energy-Based Model

Implementation for [Joint Training of Variational Auto-Encoder and Latent Energy-Based Model](https://arxiv.org/abs/2006.06059).

Pretrained Model can be found at xxx
(perhaps later you could provide the demo code: using the saved model to get fid/mse, ood)

Train on Cifar10
python train_cifar.py 

Note: the codes was tested on Unbuntu with GPU V100 and rtx 2070 super. Other platforms may/may not have numerical instablities. The fid/mse can be relatively stable while the latent EBM is not (as stated in the paper), we tend to develop the robust ebm learning in the future work. 