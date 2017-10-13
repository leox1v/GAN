import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from model import Model
import pprint
from utils import *


flags = tf.app.flags
flags.DEFINE_integer("max_iter", 35, "Maximum of iterations in thousands [35]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("input_width", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [28]")
flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. [28]")

flags.DEFINE_integer("D_h1", 128, "The hidden dimension of the first layer of the Discriminator. [128]")
flags.DEFINE_integer("G_h1", 128, "The hidden dimension of the first layer of the Generator. [128]")

flags.DEFINE_float("learning_rate", 0.001, "The learning rate of the optimizer [0.001]")

flags.DEFINE_integer("z_dim", 100, "The size of latent vector z.[100]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output_dir", "out/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")


FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # Define Computational Graph and load data set
    model = Model(flags=FLAGS)
    data = load_data(FLAGS.dataset)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    setup_directories(FLAGS.output_dir, FLAGS.summaries_dir)
    i = 0
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    for it in range(FLAGS.max_iter * int(1E3)):

        X, labels = data.train.next_batch(FLAGS.batch_size)
        Z = sample_Z(FLAGS.batch_size, model.z_dim)

        _, D_loss_curr, _, G_loss_curr, _, _, summary = sess.run(
            [model.d_solver, model.D_loss, model.g_solver, model.G_loss, model.d_gradients, model.g_gradients,
             model.merged], feed_dict={model.X: X, model.Z: Z})
        train_writer.add_summary(summary, it)

        if it % 1000 == 0:
            Z_valid = sample_Z(16, model.z_dim)
            samples = sess.run(model.g_z, feed_dict={model.Z: Z_valid})

            fig = plot(samples)
            plt.savefig(FLAGS.output_dir+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)





if __name__ == '__main__':
    tf.app.run()