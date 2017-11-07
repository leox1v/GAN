import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class Model():
    def __init__(self, flags, opt="adam", learning_rate=0.001):
        self.opt = opt
        self.FLAGS = flags
        self.img_dim = self.FLAGS.input_dim
        self.z_dim = self.FLAGS.z_dim
        self.out_img_dim = self.FLAGS.input_dim

        # Placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.FLAGS.z_dim], name='Z')

        self.g_z, self.theta_g, self.theta_d, self.D_loss, self.G_loss = self.training_procedure()
        self.get_variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}

        if opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif opt == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif opt == "extragrad":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # Copy the original network
            self.g_z_c, self.theta_g_c, self.theta_d_c, self.D_loss_c, self.G_loss_c = self.training_procedure(is_copy=True)
            self.merge_weight_mat = self.merge_weights()

            self.d_solver_c, self.d_gradients_c, d_grads_vars_c = self.minimize(optimizer, self.D_loss_c,
                                                                                self.theta_d_c)
            self.g_solver_c, self.g_gradients_c, g_grads_vars_c = self.minimize(optimizer, self.G_loss_c,
                                                                                self.theta_g_c)

            # Execute only after SGD step on copied network has been done already
            self.d_grads_vars_extra = self.minimize_extra(optimizer, self.D_loss_c, self.theta_d_c, self.theta_d)
            self.g_grads_vars_extra = self.minimize_extra(optimizer, self.G_loss_c, self.theta_g_c, self.theta_g)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.d_solver, self.d_gradients, d_grads_vars = self.minimize(optimizer, self.D_loss, self.theta_d)
        self.g_solver, self.g_gradients, g_grads_vars = self.minimize(optimizer, self.G_loss, self.theta_g)


        self.merged = self.add_tboard()

    def merge_weights(self):
        assign_ops = []
        for i in range(len(self.theta_g)):
            assign_ops.append(self.theta_g_c[i].assign(self.theta_g[i]))

        for i in range(len(self.theta_d)):
            assign_ops.append(self.theta_d_c[i].assign(self.theta_d[i]))
        return assign_ops

    def minimize(self, optimizer, loss, var_list):
        grads_vars = optimizer.compute_gradients(loss, var_list=var_list)
        solver = optimizer.apply_gradients(grads_vars)
        list_of_gradmat = tf.concat([tf.reshape(gv[0], [-1]) for gv in grads_vars], 0)
        squared_gradients = tf.reduce_sum(tf.square(list_of_gradmat))
        return solver, squared_gradients, grads_vars

    def minimize_extra(self, optimizer, loss, var_list, var_list_to_apply):
        grads_vars = optimizer.compute_gradients(loss, var_list=var_list)
        g = [gv[0] for gv in grads_vars]
        grads_vars = [(g[idx], var_list_to_apply[idx]) for idx in range(len(var_list_to_apply))]
        solver = optimizer.apply_gradients(grads_vars)
        return solver


    def training_procedure(self, is_copy=False):
        # Training Procedure
        g_z, theta_g = self.generator(self.Z, is_copy=is_copy)
        d_logit_real, theta_d = self.discriminator(self.X, is_copy=is_copy)
        d_logit_synth, _ = self.discriminator(g_z, reuse=True, is_copy=is_copy)

        # Loss for Discriminator: - log D(x) - log(1-D(G(z)))
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.zeros_like(d_logit_synth)))
        D_loss = D_loss_real + D_loss_fake

        # Loss for Generator: -log(D(G(z)))
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_synth, labels=tf.ones_like(d_logit_synth)))

        return g_z, theta_g, theta_d, D_loss, G_loss

    def generator(self, z, reuse=False, is_copy=False):
        scope = "generator"
        if is_copy:
            scope += "_copy"
        with tf.variable_scope(scope):
            g_hidden = tf.layers.dense(z, self.FLAGS.G_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="G1", reuse=reuse)
            g_z = tf.layers.dense(g_hidden, self.img_dim, kernel_initializer=xavier_initializer(), name="G2", reuse=reuse)
            if self.FLAGS.dataset == "mnist":
                g_z = tf.nn.sigmoid(g_z, name="g_z")
            else:
                g_z = tf.identity(g_z, name="g_z")
        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return g_z, theta_g


    def discriminator(self, x, reuse=False, is_copy=False):
        scope = "discriminator"
        if is_copy:
            scope += "_copy"
        with tf.variable_scope(scope):
            d_hidden = tf.layers.dense(x, self.FLAGS.D_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), reuse=reuse, name="D1")
            d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), reuse=reuse, name="D2")
        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return d_logit, theta_d

    def add_tboard(self):
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("G_SGradientSum", self.g_gradients)
        tf.summary.scalar("D_SGradientSum", self.d_gradients)
        return tf.summary.merge_all()

    def reset_graph(self):
        tf.reset_default_graph()


class ModeDiscriminator():
    "Idea based on Tong Che et al. Mode Regularized Generative Adversarial Networks (https://arxiv.org/abs/1612.02136)"
    def __init__(self, flags, tests_per_mode=10):
        self.FLAGS = flags
        self.img_dim = self.FLAGS.input_dim
        self.FLAGS = flags
        self.tests_per_mode = tests_per_mode
        self.modes = self.FLAGS.modes
        if self.FLAGS.dataset == "mnist":
            self.modes = 10

        # Placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_dim], name='X')
        std = 0.01
        noise = tf.random_normal(shape=tf.shape(self.X), mean=0.0, stddev=std, dtype=tf.float32)
        self.X += noise
        # label 1 means real data sample and 0 generated
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')

        self.train_graph()
        self.test_graph()

    def train_graph(self):
        self.d_logit, self.theta = self.discriminator(self.X)
        self.d_logit = tf.reshape(self.d_logit, [-1])

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit, labels=self.label)
        self.solver = tf.train.AdamOptimizer().minimize(loss)

    def test_graph(self):
        # At test time we give the discriminator 10 [tests_per_mode] real samples of each of the modes. One batch corresponds therefore to 10x10 images.
        self.d_res = tf.nn.sigmoid(self.d_logit)
        self.d_res = tf.reshape(self.d_res, [self.modes, self.tests_per_mode])
        self.d_res_mean = tf.reduce_mean(self.d_res, axis=1)
        threshold = 0.7
        self.missing_modes = tf.where(tf.greater(self.d_res_mean, threshold))


    def discriminator(self, x):
        with tf.variable_scope("mode_discriminator"):
            d_hidden = tf.layers.dense(x, self.FLAGS.D_h1, activation=tf.nn.relu, kernel_initializer=xavier_initializer(), name="D1")
            d_logit = tf.layers.dense(d_hidden, 1, kernel_initializer=xavier_initializer(), name="D2")
        theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="mode_discriminator")
        return d_logit, theta_d

    def reset_graph(self):
        tf.reset_default_graph()

