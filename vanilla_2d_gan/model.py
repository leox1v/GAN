import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, flags):
        self.FLAGS = flags
        self.dim = self.FLAGS.input_dim
        self.z_dim = self.FLAGS.z_dim

        # Variables for Discriminator Net
        # ------------------------------------
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim], name='X')

        self.D_W1 = tf.Variable(self.xavier_init([self.dim, self.FLAGS.D_h1]), name='D_W1')
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.FLAGS.D_h1]), name='D_b1')

        self.D_W2 = tf.Variable(self.xavier_init([self.FLAGS.D_h1, self.FLAGS.D_h2]), name='D_W2')
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.FLAGS.D_h2]), name='D_b2')

        self.D_W3 = tf.Variable(self.xavier_init([self.FLAGS.D_h2, 1]), name='D_W3')
        self.D_b3 = tf.Variable(tf.zeros(shape=[1]), name='D_b3')

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # Variables for Generator Net
        # ------------------------------------
        self.Z = tf.placeholder(tf.float32, shape=[None, self.FLAGS.z_dim], name='Z')

        self.G_W1 = tf.Variable(self.xavier_init([self.z_dim, self.FLAGS.G_h1]), name='G_W1')
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.FLAGS.G_h1]), name='G_b1')

        self.G_W2 = tf.Variable(self.xavier_init([self.FLAGS.G_h1, self.FLAGS.G_h2]), name='G_W2')
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.FLAGS.G_h2]), name='G_b2')

        self.G_W3 = tf.Variable(self.xavier_init([self.FLAGS.G_h2, self.dim]), name='G_W3')
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.dim]), name='G_b3')

        self.theta_G = [self.G_W1, self.G_W2,  self.G_W3, self.G_b1, self.G_b2,  self.G_b3]

        # Training Procedure
        self.G_sample = self.generator(self.Z)
        D_real, D_logit_real = self.discriminator(self.X)
        D_fake, D_logit_fake = self.discriminator(self.G_sample)

        # Loss for Discriminator: - log D(x) - log(1-D(G(x)))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss_real + D_loss_fake

        # Loss for Generator: -log(D(G(x)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        # Only update D(X)'s parameters, so var_list = theta_D
        #self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        d_optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate, self.FLAGS.opt_beta_1)
        d_gvs = d_optimizer.compute_gradients(self.D_loss, var_list=self.theta_D)
        self.D_solver = d_optimizer.apply_gradients(d_gvs)
        list_of_gradmat = tf.concat([tf.reshape(d_gv[0], [-1]) for d_gv in d_gvs],0)
        self.d_gradients = tf.reduce_sum(tf.square(list_of_gradmat))

        # Only update G(X)'s parameters, so var_list = theta_G
        #self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)
        g_optimizer = tf.train.AdamOptimizer()
        g_gvs = g_optimizer.compute_gradients(self.G_loss, var_list=self.theta_G)
        self.G_solver = g_optimizer.apply_gradients(g_gvs)
        list_of_gradmat = tf.concat([tf.reshape(g_gv[0], [-1]) for g_gv in g_gvs], 0)
        self.g_gradients = tf.reduce_sum(tf.square(list_of_gradmat))

        self.merged = self.add_tboard()

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_res = tf.matmul(G_h2, self.G_W3) + self.G_b3

        return G_res

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = tf.nn.sigmoid(D_logit)  # 1 probability that tells us if generated from p_data

        return D_prob, D_logit

    def add_tboard(self):
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("G_SGradientSum", self.g_gradients)
        tf.summary.scalar("D_SGradientSum", self.d_gradients)
        tf.summary.histogram("D_W1", self.D_W1)
        tf.summary.histogram("D_W2", self.D_W2)
        tf.summary.histogram("D_W3", self.D_W3)
        tf.summary.histogram("G_W1", self.G_W1)
        tf.summary.histogram("G_W2", self.G_W2)
        tf.summary.histogram("G_W3", self.G_W3)

        return tf.summary.merge_all()

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

