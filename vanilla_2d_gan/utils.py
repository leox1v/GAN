import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import dataset as myds
import seaborn as sns
from scipy.stats import multivariate_normal
from scipy.stats import entropy
from sklearn.cluster import KMeans
import progressbar
import pprint


def sample_Z(m, n):
    '''Gaussian prior for G(Z)'''
    return np.random.normal(size=[m,n])

def setup_directories(*dirs):
    for _dir in dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

class Helper():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.optimizer = self.FLAGS.opt_methods.split(" ")[0]
        self.local_img_counter = 0
        self.exp_dir = ""

    def setup_directories(self):
        result_dir = "results/" + self.FLAGS.dataset + "/"
        setup_directories(result_dir)

        i = 0
        self.exp_dir = result_dir + "exp_{}/".format(i)
        while os.path.exists(self.exp_dir):
            i += 1
            self.exp_dir = result_dir + "exp_{}/".format(i)
        os.makedirs(self.exp_dir)

        self.FLAGS.output_dir = self.exp_dir + self.FLAGS.output_dir
        self.FLAGS.summaries_dir = self.exp_dir + self.FLAGS.summaries_dir
        self.FLAGS.array_dir = self.exp_dir + self.FLAGS.array_dir

        setup_directories(self.FLAGS.output_dir, self.FLAGS.summaries_dir, self.FLAGS.array_dir)

        with open(self.exp_dir + "_info.txt", 'w') as f:
            f.write(self.get_info_string())

        return self.FLAGS

    def get_info_string(self):
        info = "Learning Rates:\n"
        for i, optimizer in enumerate(self.FLAGS.opt_methods.split(" ")):
            info += "{}: {}\n".format(optimizer, self.FLAGS.learning_rates.split(" ")[i])
        info += "\nNetwork:\n"
        info += "Generator: {} -> (ReLu) {} -> {}\n".format(self.FLAGS.z_dim, self.FLAGS.G_h1, self.FLAGS.input_dim)
        info += "Discriminator: {} -> (ReLu) {} -> {} (Sigmoid)\n".format(self.FLAGS.input_dim, self.FLAGS.D_h1, 1)
        return info


    def print_opt_methods(self, opt_methods):
        discr_keys = [key for key, val in opt_methods.items() if '_d' in key]
        gen_keys = [key for key, val in opt_methods.items() if '_g' in key]

        plt.figure(1, figsize=(10, 10))
        plt.subplot(211)
        plt.title("Gradientsum Discriminator")
        plt.xlabel("Iterations")
        for key in discr_keys:
            val = opt_methods[key]
            x = np.array(range(len(val))) * int(1000)
            plt.plot(x, val, label=key.split("_")[0])
        plt.legend()

        plt.subplot(212)
        plt.title("Gradientsum Generator")
        for key in gen_keys:
            val = opt_methods[key]
            x = np.array(range(len(val))) * int(1000)
            plt.plot(x, val, label=key.split("_")[0])
        plt.legend()

        plt.savefig(self.exp_dir + "Gradients.png")
        plt.show()

    def setup_progressbar(self):
        ind = [idx for (idx, x) in list(enumerate(self.FLAGS.opt_methods.split(" "))) if x == self.optimizer][0] + 1
        widgets = ['Running: ', self.optimizer.upper(), ' {}/{} '.format(ind, len(self.FLAGS.opt_methods.split(" "))),
                   progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA(), ]
        bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True, max_value=self.FLAGS.max_iter * int(1E3))
        return bar

    def validate_visually(self, sess, model, data):
        if self.FLAGS.dataset == "toy":
            # Get iterations*batch_size samples from the current generator
            iterations = 10
            samples = sess.run(model.g_z, feed_dict={model.Z: sample_Z(iterations * self.FLAGS.batch_size, model.z_dim)})

            # Compute the Jensen-Shannon Divergence between the generating distribution and the gaussian mixture fitted to the kmeans result
            jsd, km_centers = compute_JSD(data, samples, get_centers=True)
            fig = self.plot_heat(samples, data, km_centers, jsd)
            plt.savefig(self.FLAGS.output_dir + self.optimizer + '_{}.png'.format(str(self.local_img_counter).zfill(3)))
            plt.close()
        elif self.FLAGS.dataset == "mnist":
            Z_valid = sample_Z(16, model.z_dim)
            samples = sess.run(model.g_z, feed_dict={model.Z: Z_valid})

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

            plt.savefig(self.FLAGS.output_dir + self.optimizer + '_{}.png'.format(str(self.local_img_counter).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        else:
            raise ValueError('Unknown DataSet.')

        self.local_img_counter += 1

    def plot_heat(self, samples, fake_dataset, km_centers, jsd):
        fig, ax = plt.subplots()
        g = sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap='Greens', n_levels=20, ax=ax)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_facecolor(sns.color_palette('Greens', n_colors=256)[0])
        ax.scatter([mean[0] for mean in fake_dataset.means], [mean[1] for mean in fake_dataset.means], c='r', marker="D")
        ax.scatter(km_centers[:,0], km_centers[:,1], marker='o')
        ax.set_title("Step {}; JSD = {}".format(self.local_img_counter, np.round(jsd, 4)))

        fig = g
        return fig

    def batch_gradient(self, data, model, sess, opt_methods):
        X = data.next_batch(50000)
        Z = sample_Z(50000, model.z_dim)
        d_gradients, g_gradients = sess.run([model.d_gradients, model.g_gradients], feed_dict={model.X: X, model.Z: Z})

        opt_methods[self.optimizer + "_g"] = np.append(opt_methods[self.optimizer + "_g"], g_gradients)
        opt_methods[self.optimizer + "_d"] = np.append(opt_methods[self.optimizer + "_d"], d_gradients)

        save_opt_methods(opt_methods, self.FLAGS.array_dir)

        return opt_methods

def compute_JSD(dataset, samples, get_centers=False):
    min_ = -2
    max_ = 2
    steps = 50
    modes = dataset.modes

    # Perform kmeans on the samples from the generator
    kmeans = KMeans(n_clusters=modes).fit(samples)

    # Equally spaced points covering most of the domain of the distributions
    #x1, x2 = np.meshgrid(np.linspace(min_, max_, steps), np.linspace(min_, max_, steps))
    #X = np.vstack([np.ravel(x1), np.ravel(x2)]).T

    # true data generating distribution
    #p_data = dataset.get_pdf(X)

    # compute empirical distribution
    #variances = np.ones([modes, 2])
    #means = kmeans.cluster_centers_
    #cat_prob = 1.0 / modes * np.ones(modes)
    #for i in np.unique(kmeans.labels_):
    #    variances[i, :] = np.diag(np.cov(samples[kmeans.labels_ == i, :].T))
    #    cat_prob[i] = len(samples[kmeans.labels_ == i, :]) / len(samples)

    # distribution of the current generator
    #p_g = get_sample_pdf(means, variances, cat_prob, modes, X)

    #if np.float(np.sum(p_g)) * np.power(max_ - min_, 2) / np.power(steps, 2) > 0.6:
    #    jsd = entropy(p_data, 0.5*(p_data + p_g)) + entropy(p_data, 0.5*(p_data + p_g))
    #else:
    #    jsd = float("inf")

    jsd = 0

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


def load_opt_arrays(FLAGS):
    _dir = FLAGS.array_dir
    opt_methods_keys = FLAGS.opt_methods.split(" ")
    opt_methods = dict()
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    for method in opt_methods_keys:
        opt_methods[method + "_g"] = np.array([])
        opt_methods[method + "_d"] = np.array([])
        if os.path.isfile(_dir + method +"_g.npy"):
            opt_methods[method + "_g"] = np.array([]) #np.load(dir + method +"_g.npy")[5:]
        if os.path.isfile(_dir + method +"_d.npy"):
            opt_methods[method + "_d"] = np.array([]) #np.load(dir + method +"_d.npy")[5:]

    return opt_methods

def save_opt_methods(opt_methods, out_dir):
    for key in opt_methods.keys():
        np.save(out_dir + key + ".npy", opt_methods[key])


