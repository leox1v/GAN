import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

hidden_dim = 128
img_dim = 28*28
batch = 64
z_dim = 100

def main():
    ##
        # First we need to define the training operations
    ##
    # Specify the placeholders
    X = tf.placeholder(tf.float32, shape=[None, img_dim])
    Z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # Generate a fake image with the generator and the random noise vector z
    g_z, theta_g = generator(Z)

    # Use the discriminator to classify the real data X and then the fake data g_z
    d_real, theta_d = discriminator(X)
    d_fake, _ = discriminator(g_z, reuse=True)

    # Loss for Discriminator: - log D(x) - log(1-D(G(x)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real) * 0.9))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
    d_loss = d_loss_real + d_loss_fake

    # Loss for Generator: -log(D(G(x)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

    # Let's optimize the objectives
    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_d)
    g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##
        # Now we can TRAIN our model
    ##
    # Loading the MNIST data set
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    for i in range(int(200E3)):
        # We sample a batch from the MNIST train data set
        X_batch, labels = data.train.next_batch(batch)
        Z_batch = sample_z()
        _, _ = sess.run([d_optimizer, g_optimizer], feed_dict={X: X_batch, Z: Z_batch})

        if i % 1000 == 0:
            generated_imgs = sess.run(g_z, feed_dict={Z: sample_z(16)})
            plot(generated_imgs, i)

def generator(z):
    with tf.variable_scope("generator"):
        g_hidden = tf.layers.dense(z, hidden_dim, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="G1")
        g_logit = tf.layers.dense(g_hidden, img_dim, kernel_initializer=xavier_initializer(), name="G2")
        g_z = tf.nn.sigmoid(g_logit)
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    return g_z, theta_g

def discriminator(x, reuse=False):
    with tf.variable_scope("discriminator"):
        d_hidden = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), reuse=reuse, name="D1")
        d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), reuse=reuse, name="D2")
    theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    return d_logit, theta_d

def sample_z(batch=batch, z_dim=z_dim):
    return np.random.uniform(-1., 1.,  size=[batch, z_dim])

def plot(samples, i_ex):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:9]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plt.savefig('plots/{}.png'.format(str(int(i_ex/1000)).zfill(3)), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()