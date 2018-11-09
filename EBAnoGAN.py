import tensorflow as tf
from utils import *
from data_utils import *

tf.set_random_seed(777)

class EBAnoGAN:
    def __init__(self, conf, shape, depth):
        self.conf = conf
        self.data = conf.data
        self.batch_size = conf.batch_size
        self.w = shape[1]
        self.h = shape[2]
        self.c = shape[3]
        self.length = (self.w)*(self.h)*(self.c)
        self.depth = depth

    def gaussian_encoder(self, X_noised, phase):
        with tf.variable_scope("gaussian_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X_noised, self.depth[0], 5, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 5, 2, name = "conv_2"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[2], 5, 2, name = "conv_3"), phase, "bn_3"))
            net = leaky(bn(conv(net, self.depth[3], 5, 2, name = "conv_4"), phase, "bn_4"))
            net = tf.layers.flatten(net, name = "flatten")
            mean = dense(net, self.conf.n_z, name = "mean")
            std = tf.nn.softplus(dense(net, self.conf.n_z, name = "std")) + 1e-6

        return mean, std

    def bernoulli_decoder(self, Z, phase):
        with tf.variable_scope("bernoulli_decoder", reuse = tf.AUTO_REUSE):
            net = dense(Z, self.depth[3]*4*4, name = "fc_1")
            net = tf.reshape(net, [self.conf.batch_size, 4, 4, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 5, 2, name = "dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 5, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 5, 2, name="dconv_3"), phase, "bn_4"))
            logits = deconv(net,self.c, 5,2, name = "dconv_4")
            X_out = tf.nn.sigmoid(logits)

        return logits, X_out

    def EB_Variational_AutoEncoder(self, X, X_noised, phase):
        mean, std = self.gaussian_encoder(X_noised, phase) # [None,100]
        z = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        logits, X_out = self.bernoulli_decoder(z, phase)
        X_out = tf.clip_by_value(X_out, 1e-8, 1 - 1e-8)

        KL_Div = 0
        Recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = X))
        #ELBO = -CE - KL_Div, so loss = -ELOB = CE + KL_Div , tf.reduce_mean(0.5*tf.reduce_sum(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), 1))
        loss = KL_Div + Recon_loss
        '''
        Potential_loss = tf.reduce_mean(tf.square(z_center - z))
        '''
        return X_out, loss
