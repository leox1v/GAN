import numpy as np
import tensorflow as tf
import matplotlib as mt
mt.use('Agg')
import matplotlib.pyplot as plt
import os

from model import Model
import pprint
from utils import *
from dataset import DataSet


flags = tf.app.flags
flags.DEFINE_integer("max_iter", 25, "Maximum of iterations to train in thousands [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_dim", 2, "The dimension of the input samples. [2]")
flags.DEFINE_integer("modes", 4, "The number of gaussian modes. [4]")

flags.DEFINE_integer("D_h1", 128, "The hidden dimension of the first layer of the Discriminator. [128]")
flags.DEFINE_integer("D_h2", 128, "The hidden dimension of the second layer of the Discriminator. [128]")
flags.DEFINE_integer("G_h1", 128, "The hidden dimension of the first layer of the Generator. [128]")
flags.DEFINE_integer("G_h2", 128, "The hidden dimension of the second layer of the Generator. [128]")

flags.DEFINE_float("learning_rate", 1E-4, "The learning rate of the optimization. [1E-4]")
flags.DEFINE_float("opt_beta_1", 0.5, "The beta 1 value of the optimizer. [1E-4]")

flags.DEFINE_integer("z_dim", 256, "The size of latent vector z.[256]")
flags.DEFINE_string("output_dir", "out/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")


FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # specify the network
    model = Model(flags=FLAGS)

    # load data
    data = DataSet(modes=FLAGS.modes)

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # setup
    setup_directories(FLAGS.output_dir, FLAGS.summaries_dir)
    i = 0
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + 'train', sess.graph)

    for it in range(FLAGS.max_iter * int(1E3)):

        x_true_sample = data.next_batch(FLAGS.batch_size)

        _, D_loss, _, G_loss,_,_, summary  = sess.run([model.D_solver, model.D_loss, model.G_solver, model.G_loss,
                                                                 model.d_gradients, model.g_gradients, model.merged],
                                                                feed_dict={model.X: x_true_sample, model.Z: sample_Z(FLAGS.batch_size, model.z_dim)})
        train_writer.add_summary(summary, it)

        if it % 1000 == 0:
            validate_performance(sess, model, i, data)
            i += 1


def validate_performance(sess, model, i_img, dataset):
    # Get iterations*batch_size samples from the current generator
    iterations = 10
    samples = np.zeros([iterations*FLAGS.batch_size, FLAGS.input_dim])
    for i in range(iterations):
        samples[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size,:] = sess.run(model.G_sample, feed_dict={model.Z: sample_Z(FLAGS.batch_size, model.z_dim)})


    # Compute the Jensen-Shannon Divergence between the generating distribution and the gaussian mixture fitted to the kmeans result
    jsd, km_centers = compute_JSD(dataset, samples, get_centers=True)
    fig = plot_heat(samples, dataset, i_img, km_centers, jsd)
    plt.savefig(FLAGS.output_dir + '{}.png'.format(str(i_img).zfill(3)))




if __name__ == '__main__':
    tf.app.run()