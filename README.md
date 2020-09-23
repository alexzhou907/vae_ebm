# Joint Training of Variational Auto-Encoder and Latent Energy-Based Model

Implementation for [Joint Training of Variational Auto-Encoder and Latent Energy-Based Model](https://arxiv.org/abs/2006.06059).

## Pretrained Models

Pretrained models can be accessed [here](https://www.dropbox.com/s/a3xydf594fzaokl/cifar10_pretrained.rar?dl=0).

## Requirements:

Make sure the following environments are installed.

```
tensorflow-gpu=1.14.0
torchvision=0.4.0
pytorch=1.2.0
scipy=1.1.0
scikit-learn=0.21.2
Pillow=6.2.0
matplotlib=3.1.1
seaborn=0.9.0
```
The code was tested on Unbuntu with GPU V100 and RTX 2070 super. Other platforms may/may not have numerical instablities. The FID/MSE can be relatively stable while the latent EBM is not (as stated in the paper), we aim to develop robust ebm learning in future works. 


## Training on Cifar10:

```python train_cifar.py ```
