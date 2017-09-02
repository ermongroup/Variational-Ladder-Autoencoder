from vladder_large import *
from vladder_medium import *
from vladder_small import *

class VLadder(Network):
    def __init__(self, dataset, name=None, reg='kl', batch_size=100):
        Network.__init__(self, dataset, batch_size)
        if name is None or name == '':
            self.name = "vladder_%s" % dataset.name
        else:
            self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dims = self.dataset.data_dims
        self.latent_noise = False

        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        self.reg = reg
        if self.reg != 'kl' and self.reg != 'mmd':
            print("Unknown regularization, supported: kl, mmd")

        # Configurations
        if self.name == "vladder_celebA":
            self.cs = [3, 64, 128, 256, 512, 1024]
            self.ladder0_dim = 10
            self.ladder1_dim = 10
            self.ladder2_dim = 10
            self.ladder3_dim = 10
            self.num_layers = 4
            layers = LargeLayers(self)
            self.do_generate_conditional_samples = True
        elif self.name == "vladder_lsun":
            self.cs = [3, 64, 128, 256, 512, 1024]
            self.ladder0_dim = 20
            self.ladder1_dim = 20
            self.ladder2_dim = 20
            self.ladder3_dim = 40
            self.num_layers = 4
            layers = LargeLayers(self)
            self.do_generate_conditional_samples = True
        elif self.name == "vladder_svhn":
            self.cs = [3, 64, 128, 256, 1024]
            self.ladder0_dim = 5
            self.ladder1_dim = 5
            self.ladder2_dim = 5
            self.ladder3_dim = 10
            self.num_layers = 4
            layers = MediumLayers(self)
            self.do_generate_conditional_samples = True
        elif self.name == "vladder_mnist":
            self.cs = [1, 64, 128, 1024]
            self.ladder0_dim = 2
            self.ladder1_dim = 2
            self.ladder2_dim = 2
            self.num_layers = 3
            self.error_scale = 8.0
            layers = SmallLayers(self)
            self.do_generate_manifold_samples = True
        else:
            print("Unknown architecture name %s" % self.name)
            exit(-1)
        self.self = self

        self.input_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="input_placeholder")
        self.target_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="target_placeholder")

        # Define inference network
        self.regularization = 0.0
        if self.ladder0_dim > 0:
            self.iladder0_mean, self.iladder0_stddev = layers.ladder0(self.input_placeholder)
            self.iladder0_sample = self.iladder0_mean + tf.multiply(self.iladder0_stddev,
                                                               tf.random_normal([self.batch_size, self.ladder0_dim]))

            if self.reg == 'kl':
                self.ladder0_reg = tf.reduce_sum(-tf.log(self.iladder0_stddev) + 0.5 * tf.square(self.iladder0_stddev) +
                                                 0.5 * tf.square(self.iladder0_mean) - 0.5) / self.batch_size
            elif self.reg == 'mmd':
                prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder0_dim])
                self.ladder0_reg = compute_mmd(self.iladder0_sample, prior_sample)
            tf.summary.scalar("ladder0_reg", self.ladder0_reg)
            self.regularization += self.ladder0_reg

        if self.num_layers >= 2:
            self.ilatent1_hidden = layers.inference0(self.input_placeholder)
            if self.ladder1_dim > 0:
                self.iladder1_mean, self.iladder1_stddev = layers.ladder1(self.ilatent1_hidden)
                self.iladder1_sample = self.iladder1_mean + tf.multiply(self.iladder1_stddev,
                                                                   tf.random_normal([self.batch_size, self.ladder1_dim]))
                if self.reg == 'kl':
                    self.ladder1_reg = tf.reduce_sum(-tf.log(self.iladder1_stddev) + 0.5 * tf.square(self.iladder1_stddev) +
                                                     0.5 * tf.square(self.iladder1_mean) - 0.5) / self.batch_size
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder1_dim])
                    self.ladder1_reg = compute_mmd(self.iladder1_sample, prior_sample)
                tf.summary.scalar("ladder1_reg", self.ladder1_reg)
                self.regularization += self.ladder1_reg

        if self.num_layers >= 3:
            self.ilatent2_hidden = layers.inference1(self.ilatent1_hidden)
            if self.ladder2_dim > 0:
                self.iladder2_mean, self.iladder2_stddev = layers.ladder2(self.ilatent2_hidden)
                self.iladder2_sample = self.iladder2_mean + tf.multiply(self.iladder2_stddev,
                                                                   tf.random_normal([self.batch_size, self.ladder2_dim]))
                if self.reg == 'kl':
                    self.ladder2_reg = tf.reduce_sum(-tf.log(self.iladder2_stddev) + 0.5 * tf.square(self.iladder2_stddev) +
                                                     0.5 * tf.square(self.iladder2_mean) - 0.5) / self.batch_size
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder2_dim])
                    self.ladder2_reg = compute_mmd(self.iladder2_sample, prior_sample)
                tf.summary.scalar("latent2_kl", self.ladder2_reg)
                self.regularization += self.ladder2_reg

        if self.num_layers >= 4:
            self.ilatent3_hidden = layers.inference2(self.ilatent2_hidden)
            if self.ladder3_dim > 0:
                self.iladder3_mean, self.iladder3_stddev = layers.ladder3(self.ilatent3_hidden)
                self.iladder3_sample = self.iladder3_mean + tf.multiply(self.iladder3_stddev,
                                                                   tf.random_normal([self.batch_size, self.ladder3_dim]))
                if self.reg == 'kl':
                    self.ladder3_reg = tf.reduce_sum(-tf.log(self.iladder3_stddev) + 0.5 * tf.square(self.iladder3_stddev) +
                                                     0.5 * tf.square(self.iladder3_mean) - 0.5) / self.batch_size
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder3_dim])
                    self.ladder3_reg = compute_mmd(self.iladder3_sample, prior_sample)
                tf.summary.scalar("latent3_kl", self.ladder3_reg)
                self.regularization += self.ladder3_reg

        # Define generative network
        self.ladders = {}
        if self.num_layers >= 4 and self.ladder3_dim > 0:
            self.ladder3_placeholder = tf.placeholder(shape=(None, self.ladder3_dim), dtype=tf.float32, name="ladder3")
            self.ladders['ladder3'] = [self.ladder3_placeholder, self.ladder3_dim]
            tlatent3_state = layers.generative3(None, self.iladder3_sample)
            glatent3_state = layers.generative3(None, self.ladder3_placeholder, reuse=True)
        else:
            tlatent3_state, glatent3_state = None, None

        if self.num_layers >= 3 and self.ladder2_dim > 0:
            self.ladder2_placeholder = tf.placeholder(shape=(None, self.ladder2_dim), dtype=tf.float32, name="ladder2")
            self.ladders['ladder2'] = [self.ladder2_placeholder, self.ladder2_dim]
            tlatent2_state = layers.generative2(tlatent3_state, self.iladder2_sample)
            glatent2_state = layers.generative2(glatent3_state, self.ladder2_placeholder, reuse=True)
        elif tlatent3_state is not None:
            tlatent2_state = layers.generative2(tlatent3_state, None)
            glatent2_state = layers.generative2(glatent3_state, None, reuse=True)
        else:
            tlatent2_state, glatent2_state = None, None

        if self.num_layers >= 2 and self.ladder1_dim > 0:
            self.ladder1_placeholder = tf.placeholder(shape=(None, self.ladder1_dim), dtype=tf.float32, name="ladder1")
            self.ladders['ladder1'] = [self.ladder1_placeholder, self.ladder1_dim]
            tlatent1_state = layers.generative1(tlatent2_state, self.iladder1_sample)
            glatent1_state = layers.generative1(glatent2_state, self.ladder1_placeholder, reuse=True)
        elif tlatent2_state is not None:
            tlatent1_state = layers.generative1(tlatent2_state, None)
            glatent1_state = layers.generative1(glatent2_state, None, reuse=True)
        else:
            tlatent1_state, glatent1_state = None, None

        if self.ladder0_dim > 0:
            self.ladder0_placeholder = tf.placeholder(shape=(None, self.ladder0_dim), dtype=tf.float32, name="ladder0")
            self.ladders['ladder0'] = [self.ladder0_placeholder, self.ladder0_dim]
            self.toutput = layers.generative0(tlatent1_state, self.iladder0_sample)
            self.goutput = layers.generative0(glatent1_state, self.ladder0_placeholder, reuse=True)
        elif tlatent1_state is not None:
            self.toutput = layers.generative0(tlatent1_state, None)
            self.goutput = layers.generative0(glatent1_state, None, reuse=True)
        else:
            print("Error: no active ladder")
            exit(0)

        # Loss and training operators
        self.reconstruction_loss = tf.reduce_mean(tf.square(self.toutput - self.target_placeholder))

        self.reg_coeff = tf.placeholder_with_default(1.0, shape=[], name="regularization_coeff")

        if self.reg == 'kl':
            self.reconstruction_loss *= 16.0 * np.prod(self.data_dims)
            self.loss = self.reg_coeff * self.regularization + self.reconstruction_loss
        elif self.reg == 'mmd':
            self.regularization *= 100
            self.reconstruction_loss *= 100
            self.loss = self.regularization + self.reconstruction_loss

        tf.summary.scalar("reconstruction_loss", self.reconstruction_loss)
        tf.summary.scalar("regularization_loss", self.regularization)
        tf.summary.scalar("loss", self.loss)

        self.merged_summary = tf.summary.merge_all()
        self.iteration = 0
        self.train_op = tf.train.AdamOptimizer(0.0002).minimize(self.loss)

        # Set restart=True to not ignore previous checkpoint and restart training
        self.init_network(restart=True)
        self.print_network()
        # Set read_only=True to not overwrite previous checkpoint
        self.read_only = False

    def train(self, batch_input, batch_target, label=None):
        self.iteration += 1
        feed_dict = {
            self.input_placeholder: batch_input,
            self.reg_coeff: 1 - math.exp(-self.iteration / 15000.0),
            self.target_placeholder: batch_target
        }
        _, recon_loss, reg_loss = self.sess.run([self.train_op, self.reconstruction_loss, self.regularization],
                                                feed_dict=feed_dict)
        if self.iteration % 2000 == 0:
            self.save_network()
        if self.iteration % 20 == 0:
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.iteration)
        return recon_loss, reg_loss

    def test(self, batch_input, label=None):
        train_return = self.sess.run(self.toutput,
                                     feed_dict={self.input_placeholder: batch_input})
        return train_return

    def random_latent_code(self):
        return {key: np.random.normal(size=[self.ladders[key][1]]) for key in self.ladders}

    def generate_conditional_samples(self, condition_layer, condition_code):
        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}

        # To avoid breaking batch normalization the fixed codes must be inserted at random locations
        random_indexes = np.random.choice(range(self.batch_size), size=8, replace=False)
        for key in codes:
            if condition_layer != key:
                codes[key][random_indexes] = condition_code[key]

        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output[random_indexes], condition_code

    def generate_samples(self):
        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output

    def generate_manifold_samples(self, external_layer, external_code):
        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}

        # To avoid breaking batch normalization fixed code must be inserted at random locations
        num_insertions = 8
        if num_insertions > external_code.shape[0]:
            num_insertions = external_code.shape[0]
        random_indexes = np.random.choice(range(self.batch_size), size=num_insertions, replace=False)
        codes[external_layer][random_indexes] = external_code[0:num_insertions]
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output[random_indexes]