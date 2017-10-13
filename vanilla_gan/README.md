# Vanilla GAN on MNIST Dataset
## Run code
``` bash
git clone https://github.com/leox1v/GAN
cd GAN/vanilla_gan/
python3 main.py # oprional add flags e.g. --batch_size=32 --max_iter=20
```
## How does it work?
The implementation is based on the original paper about [General Adversarial Networks](https://arxiv.org/abs/1406.2661) by Ian Goodfellow.
<center>
![Generator](/res/Generator.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
![Discriminator](/res/Discriminator.png)
</center>

## Results

<center>
### 1k Iterations
![After 1k Iterations](/res/001.png)
### 5k Iterations
![After 5k Iterations](/res/005.png)
### 30k Iterations
![After 30k Iterations](/res/030.png)
</center>
