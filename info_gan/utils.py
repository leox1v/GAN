import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def load_data(name):
    if name == "mnist":
        return input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Add different data sets later

def sample_latent_vec(m, n, for_valid=False):
    if for_valid:
        z = np.random.uniform(-1., 1., size=[16, n - 10])
        c_mn = np.zeros([16, 10])
        c_mn[:, 1] = 1

        if m == 100:
            z = np.random.uniform(-1., 1., size=[m, n - 10])
            c_mn = np.zeros([100, 10])
            for i in range(10):
                c_mn[i*10: (i+1)*10, i] = 1
            print(c_mn)
    else:
        z = np.random.uniform(-1., 1., size=[m, n - 10])
        c_mn = np.random.multinomial(1, [.1]*10, size=m)
    lat = np.hstack([z, c_mn])
    return lat

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def info_plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def setup_directories(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)