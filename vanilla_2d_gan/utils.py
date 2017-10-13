import matplotlib as mt
mt.use('Agg')
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import dataset as myds
import seaborn as sns
from scipy.stats import multivariate_normal
from scipy.stats import entropy
from sklearn.cluster import KMeans


def sample_Z(m, n):
    '''Gaussian prior for G(Z)'''
    return np.random.normal(size=[m,n])


def plot_heat(samples, fake_dataset, global_step, km_centers, jsd):
    fig, ax = mt.pyplot.subplots()
    g = sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap='Greens', n_levels=20, ax=ax)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_facecolor(sns.color_palette('Greens', n_colors=256)[0])
    ax.scatter([mean[0] for mean in fake_dataset.means], [mean[1] for mean in fake_dataset.means], c='r', marker="D")
    ax.scatter(km_centers[:,0], km_centers[:,1], marker='o')
    ax.set_title("Step {}; JSD = {}".format(global_step, np.round(jsd, 4)))

    fig = g
    return fig

def compute_JSD(dataset, samples, get_centers=False):
    min_ = -2
    max_ = 2
    steps = 50
    modes = dataset.modes

    # Perform kmeans on the samples from the generator
    kmeans = KMeans(n_clusters=modes).fit(samples)

    # Equally spaced points covering most of the domain of the distributions
    x1, x2 = np.meshgrid(np.linspace(min_, max_, steps), np.linspace(min_, max_, steps))
    X = np.vstack([np.ravel(x1), np.ravel(x2)]).T

    # true data generating distribution
    p_data = dataset.get_pdf(X)

    # compute empirical distribution
    variances = np.ones([modes, 2])
    means = kmeans.cluster_centers_
    cat_prob = 1.0 / modes * np.ones(modes)
    for i in np.unique(kmeans.labels_):
        variances[i, :] = np.diag(np.cov(samples[kmeans.labels_ == i, :].T))
        cat_prob[i] = len(samples[kmeans.labels_ == i, :]) / len(samples)

    # distribution of the current generator
    p_g = get_sample_pdf(means, variances, cat_prob, modes, X)

    if np.float(np.sum(p_g)) * np.power(max_ - min_, 2) / np.power(steps, 2) > 0.6:
        jsd = entropy(p_data, 0.5*(p_data + p_g)) + entropy(p_data, 0.5*(p_data + p_g))
    else:
        jsd = float("inf")

    if get_centers:
        return jsd, kmeans.cluster_centers_
    else:
        return jsd


def get_sample_pdf(means, variances, cat_probs, modes, X):
    sums = np.zeros(np.array(X).shape[0])
    for n, x in enumerate(X):
        for i in range(modes):
            sums[n] += cat_probs[i] * multivariate_normal.pdf(x, means[i], variances[i])
    return sums

def setup_directories(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

