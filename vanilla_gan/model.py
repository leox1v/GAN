import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class Model():
    def __init__(self, flags):
        self.FLAGS = flags
        self.img_dim = self.FLAGS.input_height * self.FLAGS.input_width
        self.z_dim = self.FLAGS.z_dim
        self.out_img_dim = self.FLAGS.output_height * self.FLAGS.output_width

        # Placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.FLAGS.z_dim], name='Z')

        # Training Procedure
        self.g_z, self.theta_g = self.generator(self.Z)
        d_logit_real, self.theta_d = self.discriminator(self.X)
        d_logit_synth, _ = self.discriminator(self.g_z, reuse=True)

        # Loss for Discriminator: - log D(x) - log(1-D(G(z)))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.zeros_like(d_logit_synth)))
        self.D_loss = D_loss_real + D_loss_fake

        # Loss for Generator: -log(D(G(z)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.ones_like(d_logit_synth)))

       # Optimize the loss functions and get the square gradients
        optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate)
        self.d_solver, self.d_gradients = self.minimize(optimizer, self.D_loss, self.theta_d)
        self.g_solver, self.g_gradients = self.minimize(optimizer, self.G_loss, self.theta_g)

        self.merged = self.add_tboard()

    def minimize(self, optimizer, loss, var_list):
        grads_vars = optimizer.compute_gradients(loss, var_list=var_list)
        solver = optimizer.apply_gradients(grads_vars)
        list_of_gradmat = tf.concat([tf.reshape(gv[0], [-1]) for gv in grads_vars], 0)
        squared_gradients = tf.reduce_sum(tf.square(list_of_gradmat))
        return solver, squared_gradients

    def generator(self, z):
        with tf.variable_scope("generator"):
            g_hidden = tf.layers.dense(z, self.FLAGS.G_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="G1")
            g_logit = tf.layers.dense(g_hidden, self.img_dim, kernel_initializer=xavier_initializer(), name="G2")
            g_z = tf.nn.sigmoid(g_logit)
        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return g_z, theta_g


    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator"):
            d_hidden = tf.layers.dense(x, self.FLAGS.D_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), reuse=reuse, name="D1")
            d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), reuse=reuse, name="D2")
        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return d_logit, theta_d

    def add_tboard(self):
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("G_SGradientSum", self.g_gradients)
        tf.summary.scalar("D_SGradientSum", self.d_gradients)
        gen_img = tf.reshape(self.g_z, [-1, 28, 28, 1])
        tf.summary.image("Generated_img", gen_img)
        return tf.summary.merge_all()


