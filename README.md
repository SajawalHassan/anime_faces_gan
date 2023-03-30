# <h1 align="center">Anime_faces_gan</h1>
This repository contains a Generative Adversarial Network (GAN) implementation to generate images of anime faces.

## Overview
The GAN consists of two neural networks: a generator and a discriminator. The generator takes a latent vector of size (x, 128 (latent size), y, z) as input and generates an image of size 3x64x64 as output. The discriminator takes an image of size 3x64x64 as input and outputs a scalar value indicating whether the image is real or fake.

## Requirements
* PyTorch (version >= 1.9.0)
* TorchVision
* NumPy
* matplotlib
* tqdm
* MatplotLib

## Usage
This GAN can be used to generate images of anime faces, you give it a 4D tensor with size=(x, 128 (latent size), y, z) and it spits out an image of size=(3,64,64) which somewhat resembles an anime face.

## Generator
The Generator is given a 4D tensor of size (x, 128 (latent size), y, z) and it gives us an image using that tensor. It works by having multiple sets of Transposed Convolution Layers, followed by a BatchNormalization layer and having a ReLU. Here is the architecture:

```
nn.ConvTranspose2d(LATENT_SIZE, 512, kernel_size=4, stride=1, padding=0, bias=False),
nn.BatchNorm2d(512),
nn.ReLU(True),
# out: 512 x 4 x 4

nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(256),
nn.ReLU(True),
# out: 256 x 8 x 8

nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(128),
nn.ReLU(True),
# out: 128 x 16 x 16

nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(64),
nn.ReLU(True),
# out: 64 x 32 x 32

nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
nn.Tanh()
# out: 3 x 64 x 64
```

## Discriminator
The discriminator takes a tensor of size (3, 64, 64) and gives a scaler value telling us wheather the image is real or fake
```
nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(64),
nn.LeakyReLU(0.2, inplace=True),
# out: 64 x 32 x 32

nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(128),
nn.LeakyReLU(0.2, inplace=True),
# out: 128 x 16 x 16

nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(256),
nn.LeakyReLU(0.2, inplace=True),
# out: 256 x 8 x 8

nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
nn.BatchNorm2d(512),
nn.LeakyReLU(0.2, inplace=True),
# out: 512 x 4 x 4

nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
# out: 1 x 1 x 1

nn.Flatten(),
nn.Sigmoid()
```

## Acknoledgements
The dataset used here is the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) which consists of 63,000 images of anime faces.
