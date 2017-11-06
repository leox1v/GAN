import numpy as np
import collections
from scipy.stats import multivariate_normal
from tensorflow.examples.tutorials.mnist import input_data

class DataSet():
    def __init__(self, dataset,radius=1.0, std=0.01, modes=8, dim=2):
        self.dataset = dataset
        if self.dataset == "toy":
            self.modes = modes
            self.dim = dim
            self.radius = radius
            self.std = std
            self.means = self.generate_fake_distribution()
            self.sample_data = self.next_batch(10000)
        elif self.dataset == "mnist":
            self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def generate_fake_distribution(self):
        thetas = np.linspace(0, 2 * np.pi, self.modes+1)
        xs, ys = self.radius * np.sin(thetas[:self.modes]), self.radius * np.cos(thetas[:self.modes])
        loc = np.vstack([xs,ys]).T
        return loc

    def get_pdf(self, X):
        """
        Function to query the probability density function of the mixture of gaussians.
        :param X: with shape (n_samples, 2). Points for which we want to query the pdf function of the mixture.
        :return: Array of size n_samples with the function values for X.
        """
        sums = np.zeros(np.array(X).shape[0])
        for n, x in enumerate(X):
            for i in range(self.modes):
                sums[n] += 1.0 / self.modes * multivariate_normal.pdf(x, self.means[i], self.std)
        return sums


    def next_batch(self, batch_size, test=False):
        if self.dataset == "toy":
            rand_modes = np.random.choice(self.modes, batch_size)
            mean = np.array([self.means[mode] for mode in rand_modes])
            samples = np.zeros([batch_size, self.dim])
            for i in range(batch_size):
                samples[i, :] = np.random.multivariate_normal(mean[i], [[self.std,0], [0,self.std]])
            return samples
        elif self.dataset == "mnist":
            if test:
                samples, labels = self.data.test.next_batch(batch_size)
                return samples, labels
            else:
                samples, labels = self.data.train.next_batch(batch_size)
                return samples


