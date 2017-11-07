import numpy as np
import tensorflow as tf
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from model import Model
import pprint
from utils import *
from utils import Helper
from dataset import DataSet
import progressbar


flags = tf.app.flags
flags.DEFINE_integer("max_iter", 10, "Maximum of iterations to train in thousands [25]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [64]")

flags.DEFINE_string("dataset", "toy", "The dataset that is used. [toy, mnist]")
flags.DEFINE_integer("input_dim", 2, "The dimension of the input samples. [2]")

flags.DEFINE_integer("modes", 4, "The number of gaussian modes. [4]")

flags.DEFINE_integer("D_h1", 10, "The hidden dimension of the first layer of the Discriminator. [10]")
flags.DEFINE_integer("G_h1", 10, "The hidden dimension of the first layer of the Generator. [10]")

flags.DEFINE_integer("z_dim", 10, "The size of latent vector z.[256]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output_dir", "out/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("summaries_dir", "tensorboard/", "Directory to use for the summary.")
flags.DEFINE_string("array_dir", "arr/", "Directory to use for arrays to store.")
flags.DEFINE_string("opt_methods", "extragrad sgd adagrad adam", "Optimization methods that needs to be compared")
flags.DEFINE_string("learning_rates", "0.05 0.01 0.1 0.001", "Learning rates for the different opt_methods, respectively.")


pp = pprint.PrettyPrinter()

def main(_):
    FLAGS = flags.FLAGS
    pp.pprint(flags.FLAGS.__flags)

    # load data
    data = DataSet(dataset=FLAGS.dataset, modes=FLAGS.modes)
    helper = Helper(FLAGS)
    FLAGS = helper.setup_directories()

    opt_methods = load_opt_arrays(FLAGS)

    for i, opt in enumerate(FLAGS.opt_methods.split(" ")):
        learning_rate = float(FLAGS.learning_rates.split(" ")[i])
        opt_methods = train(opt, opt_methods, data, helper, FLAGS, learning_rate)

    helper.print_opt_methods(opt_methods)



def train(optimizer, opt_methods, data, helper, FLAGS, learning_rate):
    # specify the network
    model = Model(flags=FLAGS, opt=optimizer, learning_rate=learning_rate)
    helper.optimizer = optimizer

    # initialize session
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    modelname = "model_{}".format(optimizer)
    if os.path.isfile(FLAGS.checkpoint_dir + modelname + ".meta"):
        saver.restore(sess, FLAGS.checkpoint_dir + modelname)
        print("Model loaded from: '{}'.".format(FLAGS.checkpoint_dir + modelname))
    else:
        print("\n[i] New model initialized.")


    # setup
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + 'train', sess.graph)
    bar = helper.setup_progressbar()

    helper.local_img_counter = 0

    for it in range(FLAGS.max_iter * int(1E3) + 1):

        # Data batch
        X = data.next_batch(FLAGS.batch_size)
        Z = sample_Z(FLAGS.batch_size, model.z_dim)

        if optimizer == "extragrad":
            # Set weights from copied network to the original one
            sess.run([model.merge_weight_mat])

            # Do SGD step on the copied network
            sess.run([model.d_solver_c, model.g_solver_c], feed_dict={model.X: X, model.Z: Z})

            # Do the extragradient step where I compute the gradients on the updated copied network and then apply those gradients
            # to the original network
            _,_, summary = sess.run([model.d_grads_vars_extra, model.g_grads_vars_extra, model.merged], feed_dict={model.X: X, model.Z: Z})

        else:
           _, D_loss, _, G_loss,_,_, summary  = sess.run([model.d_solver, model.D_loss, model.g_solver, model.G_loss,
                                                                     model.d_gradients, model.g_gradients, model.merged],
                                                                    feed_dict={model.X: X, model.Z: Z})

        train_writer.add_summary(summary, it)


        if it % 1000 == 0:
            bar.update(it)

            # Validation
            helper.validate_visually(sess, model, data)

            # Get the gradient of the whole training set and add the value to the opt_arrays
            opt_methods = helper.batch_gradient(data, model, sess, opt_methods)

    bar.finish()
    saver.save(sess, FLAGS.checkpoint_dir + modelname)
    model.reset_graph()
    return opt_methods



if __name__ == '__main__':
    tf.app.run()