import numpy as np

from dataset import DataSet
from model import ModeDiscriminator, Model
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

# Parameters
dataset = "toy"
exp_no = 0


def main():
    # Parameters
    train_iterations = 5000
    batch_size = 64
    try:
        FLAGS = get_flags(dataset, exp_no)
    except:
        raise AttributeError("Flags have not been saved. So the parameters are lost.")

    checkpoint_dir = FLAGS.checkpoint_dir

    results = dict()
    data = DataSet(dataset=FLAGS.dataset, modes=FLAGS.modes)

    tests_per_mode = 20
    sorted_batch, sorted_labels = data.sorted_batch(tests_per_mode)

    for optimizer in FLAGS.opt_methods.split(" "):

        modelname = "model_{}".format(optimizer)

        model = ModeDiscriminator(FLAGS, tests_per_mode)

        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(checkpoint_dir + modelname + '.meta')
        sess.run(tf.global_variables_initializer())
        new_saver.restore(sess, checkpoint_dir + modelname)
        #names = [n.name for n in tf.get_default_graph().as_graph_def().node]

        for i in range(train_iterations):
            # Get a batch from the generator network
            z_gen = sample_Z(int(batch_size/2), FLAGS.z_dim)
            samples_gen = sess.run(['generator/g_z:0'], feed_dict={'Z:0': z_gen})
            samples_gen = np.reshape(samples_gen, [int(batch_size/2), -1])

            # Get a batch from the real data
            X = data.next_batch(int(batch_size/2))

            # Stack and shuffle the batch
            samples = np.vstack([samples_gen, X])
            labels = np.hstack([np.zeros(int(batch_size/2)), np.ones(int(batch_size/2))])
            perm = np.random.permutation(len(labels))
            samples = samples[perm, :]
            labels = labels[perm]

            # Train the new (mode) discriminator model. Half of the training examples are samples from the generator and half are real data samples
            sess.run(model.solver, feed_dict={model.X: samples, model.label: labels})

        # Test the trained network and specify the missing modes
        d_res_mean, d_res, missing_modes = sess.run([model.d_res_mean, model.d_res, model.missing_modes],
                                                    feed_dict={model.X: sorted_batch, model.label: sorted_labels})
        print("----------------- \n Results for {}-model".format(optimizer.upper()))
        print("Per Mode mean confidence: {}".format(d_res_mean))
        print("Missing modes: {}".format(np.ravel(missing_modes)))


        results[optimizer + "_per_mode"] = np.array(np.array(d_res_mean) * 1000).astype(int)/ 1000.0
        results[optimizer + "_missing_modes"] = np.ravel(missing_modes)

        model.reset_graph()

    opt_methods = load_opt_arrays(FLAGS)
    Helper(FLAGS).print_opt_methods(opt_methods, results, img_name="Gradients_annotated")


    return results











if __name__ == '__main__':
    main()