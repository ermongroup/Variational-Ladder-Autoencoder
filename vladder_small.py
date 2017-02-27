from abstract_network import *


class SmallLayers:
    """ Definition of layers for a medium sized ladder network """
    def __init__(self, network):
        self.network = network

    def inference0(self, input_x):
        with tf.variable_scope("inference0"):
            conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4, 4], 2)
            conv2 = conv2d_bn_lrelu(conv1, self.network.cs[2], [4, 4], 2)
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1 = tf.contrib.layers.fully_connected(conv2, self.network.cs[3], activation_fn=tf.identity)
            return fc1

    def ladder0(self, input_x):
        with tf.variable_scope("ladder0"):
            conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4, 4], 2)
            conv2 = conv2d_bn_lrelu(conv1, self.network.cs[2], [4, 4], 2)
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.identity)
            fc1_stddev = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.sigmoid)
            return fc1_mean, fc1_stddev

    def inference1(self, latent1):
        with tf.variable_scope("inference1"):
            fc1 = fc_bn_lrelu(latent1, self.network.cs[3])
            fc2 = fc_bn_lrelu(fc1, self.network.cs[3])
            return fc2

    def ladder1(self, latent1):
        with tf.variable_scope("ladder1"):
            fc1 = fc_bn_lrelu(latent1, self.network.cs[3])
            fc2 = fc_bn_lrelu(fc1, self.network.cs[3])
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder1_dim, activation_fn=tf.identity)
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder1_dim, activation_fn=tf.sigmoid)
            return fc3_mean, fc3_stddev

    def ladder2(self, latent1):
        with tf.variable_scope("ladder2"):
            fc1 = fc_bn_lrelu(latent1, self.network.cs[3])
            fc2 = fc_bn_lrelu(fc1, self.network.cs[3])
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder2_dim, activation_fn=tf.identity)
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder2_dim, activation_fn=tf.sigmoid)
            return fc3_mean, fc3_stddev

    def combine_noise(self, latent, ladder, method='gated_add', name="default"):
        if method is 'concat':
            return tf.concat(len(latent.get_shape())-1, [latent, ladder])
        else:
            if method is 'add':
                return latent + ladder
            elif method is 'gated_add':
                gate = tf.get_variable("gate", shape=latent.get_shape()[1:], initializer=tf.constant_initializer(0.1))
                tf.summary.histogram(name + "_noise_gate", gate)
                return latent + tf.mul(gate, ladder)

    def generative0(self, latent1, ladder0=None, reuse=False):
        with tf.variable_scope("generative0") as gs:
            if reuse:
                gs.reuse_variables()

            if ladder0 is not None:
                ladder0 = fc_bn_lrelu(ladder0, self.network.cs[3])
                if latent1 is not None:
                    latent1 = self.combine_noise(latent1, ladder0, name="generative0")
                else:
                    latent1 = ladder0
            elif latent1 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent1, int(self.network.fs[2] * self.network.fs[2] * self.network.cs[2]))
            fc1 = tf.reshape(fc1, tf.pack([tf.shape(fc1)[0], self.network.fs[2], self.network.fs[2], self.network.cs[2]]))
            conv1 = conv2d_t_bn_relu(fc1, self.network.cs[1], [4, 4], 2)
            output = tf.contrib.layers.convolution2d_transpose(conv1, self.network.data_dims[-1], [4, 4], 2,
                                                               activation_fn=tf.sigmoid)
            output = (self.network.dataset.range[1] - self.network.dataset.range[0]) * output + self.network.dataset.range[0]
            return output

    def generative1(self, latent2, ladder1=None, reuse=False):
        with tf.variable_scope("generative1") as gs:
            if reuse:
                gs.reuse_variables()
            if ladder1 is not None:
                ladder1 = fc_bn_relu(ladder1, self.network.cs[3])
                if latent2 is not None:
                    latent2 = self.combine_noise(latent2, ladder1, name="generative1")
                else:
                    latent2 = ladder1
            elif latent2 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent2, self.network.cs[3])
            fc2 = fc_bn_relu(fc1, self.network.cs[3])
            fc3 = tf.contrib.layers.fully_connected(fc2, self.network.cs[3], activation_fn=tf.identity)
            return fc3

    def generative2(self, latent3, ladder2, reuse=False):
        with tf.variable_scope("generative2") as gs:
            if reuse:
                gs.reuse_variables()
            fc1 = fc_bn_relu(ladder2, self.network.cs[3])
            fc2 = fc_bn_relu(fc1, self.network.cs[3])
            fc3 = tf.contrib.layers.fully_connected(fc2, self.network.cs[3], activation_fn=tf.identity)
            return fc3

