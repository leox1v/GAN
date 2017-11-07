# Vanilla GAN implementation on MNIST or 2D-toy data with different optimization schemes
## Run code
``` bash
git clone https://github.com/leox1v/GAN
cd GAN/gan_framework/
python3 main.py # oprional add flags e.g. --dataset="mnist" --opt_methods="adam sgd extragrad adagrad"

# To run the mode counting network that evaluates mode collapsing for the specified experiment 
python3 mode_counting.py
```

## High level Description
Here we train a simple vanilla GAN with only one hidden layer with a ReLu activation in both the generator and the disciriminator. The number of hidden units as well as the input dimension for the latent code needs to be specifies via the app flags.
The goal of this framework is to compare different optimization schemes. The implemented methods are: adam, sgd, extragrad, adagrad. In order to find saddle points with vanishing derivatives, we evaluate the the gradients of the generator and discriminator at the (whole) dataset every 1000th iteration in the training.
In the end a plot is produced that shows the evaluated gradients over time for the selected optimizers.


## Optimizers

![Equations](/imgs/optimizers_equ.png).

<!--### SGD
$$ \Theta^{t+1} \leftarrow \Theta^t - \eta \frac{1}{n}\sum_{i=1}^n \nabla_{\Theta} J(\Theta^t; x_n, y_n) $$
### AdaGrad
AdaGrad adapts the learning rate. It sums up the gradients of the previous steps for all the parameters. The larger the sum for a specific parameter the more it has been updated already and the learning rate is chosen to be a smaller value.

$$  \begin{align} g^t_i &= \nabla_{\Theta_i} J(\Theta^t) \\
					G_{i,i} &= \sum_{\tau=0}^t (g^\tau_i)^2 \\
					\Theta^{t+1}_i &\leftarrow \Theta^{t}_i - \frac{\eta}{\sqrt{G_i,i}}\, g^t_i 
\end{align}$$  
					
Since the denominator of the update factor is the l2-norm of previous derivatives, extreme parameter updates get dampened, while parameters that get few or small updates receive higher learning rates.
### ExtraGrad
Extragradient method does an extrapolation step for the evaluation of the gradient at the next iteration.

$$  \begin{align} \Theta^{t+1/2} &\leftarrow \Theta^t - \eta \nabla_{\Theta}J(\Theta^t) \\
					 \Theta^{t+1} &\leftarrow \Theta^t - \eta \nabla_{\Theta}J(\Theta^{t+1/2})
\end{align}$$ 

### Adam
Adam also computes adptive learning rates for each parameter. Additionally, it also considers past gradients, similar to momentum.

$$  \begin{align} m^{t+1}_i &= \beta_1 m^{t}_i + (1- \beta_1) \nabla_{\Theta_i}J(\Theta^t) \\
					 v^{t+1}_i &= \beta_2 v^{t}_i + (1- \beta_2) (\nabla_{\Theta_i}J(\Theta^t))^2 \\
					 \hat{m_i} &= \frac{m^{t+1}_i}{1 - \beta_1} \\
 					 \hat{v_i} &= \frac{v^{t+1}_i}{1 - \beta_2} \\
 					 \Theta^{t+1}_i &\leftarrow \Theta^{t}_i - \eta \frac{\hat{m_i}}{\sqrt{\hat{v_i}} + \epsilon}
\end{align}$$ 
-->

## Mode Collapsing
Mode Collapsing is a known issue of GANs addressed in many papers []. The question is whether mode collapsing is related to non-optimal saddle-points of the objective function, i.e. do we have an equlibrium with vanishing gradients in the collapsed state. If those saddle points exists, which optimization schemes are best in overcoming those and leading to a global optimum?
### Measuring Mode Collapsing
Since it is not clear how to measure mode collapsing in a high dimensional settings, e.g. images, we use the following technique proposed by Tong Che et al..

After the full training of the GAN, we train a new discriminator D* to distinguish between samples from the generator and the real data. In order to determine "missing modes" we test D* after training on the test set T of the real dataset. If for test samples $t \in T$ with the same label the discrimination value $D(t)$ is close to 1, we can conclude that the corresponding mode is missing. For this technique we assume we know the number of modes in the dataset. For example, for the mnist data set it is reasonable to assume 10 modes corresponding to the 10 different labels.  



## Results