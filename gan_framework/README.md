# Vanilla GAN implementation on MNIST or 2D-toy data with different optimization schemes
## Run code
``` bash
git clone https://github.com/leox1v/GAN
cd GAN/vanilla_2d_gan/
python3 main.py # oprional add flags e.g. --dataset="mnist" --opt_methods="adam sgd extragrad adagrad"
```

## High level Description
This trains a simple vanilla GAN with one hidden ReLu layer in generator and discriminator with the specified number of units. Different optimization methods (adam, sgd, extragrad, adagrad) can be chosen to be compared. After every 1000th iteration the gradients of the generator and the discriminator are evaluated at the (whole) training set to find equilibrium points at which both gradients go to zero. A plot is produced that shows the evaluated gradients over time for the different optimizers.


## Optimizers

### SGD
### AdaGrad
### ExtraGrad
### Adam

## Mode Collapsing
Mode Collapsing is a known issue of GANs addressed in many papers []. The question is whether mode collapsing is related to non-optimal saddle-points of the objective function. And if those exist, which optimization schemes can overcome these points and why. TODO

### Measuring Mode Collapsing
TODO

## Results
