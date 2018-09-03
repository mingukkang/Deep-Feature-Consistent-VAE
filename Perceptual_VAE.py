import tensorflow as tf
from utils import *
from data_utils import *
from keras import applications

class Perceptual_VAE:
    def __init__(self, conf, shape):
        self.conf = conf
        self.data = conf.data
        self.batch_size = conf.batch_size
        self.w = shape[1]
        self.h = shape[2]
        self.c = shape[3]
        self.length = (self.w)*(self.h)*(self.c)

    def gaussian_encoder(self, X, phase):
        with tf.variable_scope("gaussian_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(bn(conv(X,32,4,2, name = "conv_1"),is_training = phase, name = "bn_1"))
            net = leaky(bn(conv(net, 64, 4, 2, name="conv_2"),is_training = phase, name = "bn_2"))
            net = leaky(bn(conv(net, 128, 4, 2, name="conv_3"),is_training = phase, name = "bn_3"))
            net = leaky(bn(conv(net, 256, 4, 2, name="conv_4"),is_training = phase, name = "bn_4"))
            net = tf.layers.flatten(net)
            mean = dense(net, self.conf.n_z, name = "mean")
            std = tf.nn.softplus(dense(net, self.conf.n_z, name = "std"))+1e-6
        return mean, std

    def bernoulli_decoder(self,Z, phase):
        with tf.variable_scope("bernoulli_decoder", reuse = tf.AUTO_REUSE):
            net = dense(Z,4096, name = "dense_1")
            net = tf.reshape(net, [self.conf.batch_size, 4, 4, 256])
            net = leaky(bn(conv(upsample(net,8), 128, 3, 1, name ="conv_1"),is_training = phase, name = "bn_1"))
            net = leaky(bn(conv(upsample(net, 16), 64, 3, 1, name="conv_2"),is_training = phase, name = "bn_2"))
            net = leaky(bn(conv(upsample(net, 32), 32, 3, 1, name="conv_3"),is_training = phase, name = "bn_3"))
            net = tf.nn.sigmoid(conv(upsample(net, 64), 3, 3, 1, name = "conv_4"))
        return net

    def Variational_AutoEncoder(self, X, phase):
        mean, std = self.gaussian_encoder(X, phase)
        z = mean + std*tf.random_normal(tf.shape(mean, out_type = tf.int32), 0, 1, dtype = tf.float32)
        output = self.bernoulli_decoder(z, phase)
        output = tf.clip_by_value(output, 1e-8, 1 - 1e-8)
        KL_Div = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mean) + tf.square(std) - tf.log(1e-8 + tf.square(std)) - 1, 1))

        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        for layer in model.layers[:]:
            layer.trainable = False

        act_1 = model.layers[0](output)
        act_2 = model.layers[1](act_1)
        act_3 = model.layers[2](act_2)

        act_X_1 = model.layers[0](X)
        act_X_2 = model.layers[1](act_X_1)
        act_X_3 = model.layers[2](act_X_2)

        p_loss_1 = tf.reduce_mean(tf.squared_difference(act_1, act_X_1))
        p_loss_2 = tf.reduce_mean(tf.squared_difference(act_2, act_X_2))
        p_loss_3 = tf.reduce_mean(tf.squared_difference(act_3, act_X_3))

        Perceptual_loss = p_loss_1 + p_loss_2 + p_loss_3

        total_loss = KL_Div + 0.5*Perceptual_loss

        return output, z,total_loss




