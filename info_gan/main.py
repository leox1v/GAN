import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from model import Model
from model_conv import ConvModel
import pprint
from utils import *
import progressbar


flags = tf.app.flags
flags.DEFINE_integer("max_iter", 50, "Maximum of iterations in thousands [35]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("input_width", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [28]")
flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. [28]")

flags.DEFINE_integer("D_h1", 128, "The hidden dimension of the first layer of the Discriminator. [128]")
flags.DEFINE_integer("G_h1", 128, "The hidden dimension of the first layer of the Generator. [128]")

flags.DEFINE_string("model", "dense", "The model to use for the generator and disctiminator [conv, dense].")

flags.DEFINE_float("learning_rate", 0.001, "The learning rate of the optimizer [0.001]")

flags.DEFINE_integer("z_dim", 100, "The size of latent vector z.[100]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist]")
flags.DEFINE_string("checkpoint_dir", "check/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output_dir", "out/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")


FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

modelname = "model_{}.ckpt".format(FLAGS.model)
output_dir = FLAGS.output_dir + FLAGS.model + "/"
summaries_dir = FLAGS.summaries_dir + FLAGS.model + "/"

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # Define Computational Graph and load data set
    if FLAGS.model == "conv":
        model = ConvModel(FLAGS)
    else:
        model = Model(FLAGS)
    data = load_data(FLAGS.dataset)

    setup_directories(output_dir, summaries_dir, FLAGS.checkpoint_dir)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if os.path.isfile(FLAGS.checkpoint_dir + modelname + ".meta"):
        saver.restore(sess, FLAGS.checkpoint_dir + modelname)
        print("Model loaded from: '{}'.".format(FLAGS.checkpoint_dir + modelname))
    else:
        print("New model initialized.")


    i = 0
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
    bar = progressbar.ProgressBar(redirect_stdout=True)

    for it in range(FLAGS.max_iter * int(1E3)):

        X, labels = data.train.next_batch(FLAGS.batch_size)
        lat = sample_latent_vec(FLAGS.batch_size, model.z_dim)

        _, D_loss_curr, _, G_loss_curr, _, _, _ = sess.run(
            [model.d_solver, model.D_loss, model.g_solver, model.G_loss, model.d_gradients, model.g_gradients, model.Q_loss], feed_dict={model.X: X, model.lat: lat})

        if it % 1000 == 0:
            summary = sess.run(model.merged, feed_dict={model.X: X, model.lat: lat})
            train_writer.add_summary(summary, it)

        if it % 1000 == 0:
            lat_valid = sample_latent_vec(16, model.z_dim, for_valid=True)
            samples = sess.run(model.g_z, feed_dict={model.lat: lat_valid})

            fig = plot(samples)
            plt.savefig(output_dir+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        if it % 1000 == 0:
            bar.update(it / (FLAGS.max_iter * int(1E3)) * 100)

        if it % 1000 == 0:
            saver.save(sess, FLAGS.checkpoint_dir + modelname)

    lat_valid = sample_latent_vec(100, model.z_dim, for_valid=True)
    samples = sess.run(model.g_z, feed_dict={model.lat: lat_valid})

    fig = info_plot(samples)
    plt.savefig(output_dir + 'final_info.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    tf.app.run()