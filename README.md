# GAN

This repository contains different GAN (https://arxiv.org/abs/1406.2661) implementations on different data sets. 

## Vanilla GAN
The standard GAN implementation with shallow fully connected layers in Generator and Discriminator. 
1. On MNIST data set
    * Can be found at [./vanilla_gan](https://github.com/leox1v/GAN/tree/master/vanilla_gan)
2. On artificial data sampled (2D mixture of gaussians)
    * Can be found at [./vanilla_2d_gan](https://github.com/leox1v/GAN/tree/master/vanilla_2d_gan)

## Info GAN
1. On MNIST data set
   * Can be found at [./info_gan](https://github.com/leox1v/GAN/tree/master/info_gan)
   * It is possible to choose between a simple network with 2 fully connected layers in generator and discriminator or with a more sophisticated approach with additional convolutional layers.
