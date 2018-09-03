import tensorflow as tf
import numpy as np
import time
from Perceptual_VAE import *
from utils import *
from plot import *
from Perceptual_VAE import *
from data_utils import *
import pdb

DEFINE_string("data", "CelebA", "[MNIST | CIFAR_10 | CelebA]")

DEFINE_integer("n_epoch", 150, "number of Epoch for training")
DEFINE_integer("n_z", 100, "Dimension of Latent variables")
DEFINE_integer("batch_size", 128, "Batch Size for training")

DEFINE_float("decay_rate", 0.95, "rate for learning rate decay")
DEFINE_float("learning_rate", 0.0005, "learning rate")

conf = print_user_flags(line_limit = 100)
print("-"*80)

data_pipeline = data_pipeline(conf.data)
train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = data_pipeline.load_preprocess_data()
shape = np.shape(train_xs)

X = tf.placeholder(tf.float32, shape = [conf.batch_size, shape[1], shape[2], shape[3]], name = "Inputs")
latent = tf.placeholder(tf.float32, shape = [conf.batch_size, conf.n_z], name = "latent_value")
phase = tf.placeholder(tf.bool, name = "training_phase")
global_step = tf.Variable(0, trainable = False, name = "global_step")

PVAE = Perceptual_VAE(conf, shape)
images_generated,z, total_loss = PVAE.Variational_AutoEncoder(X, phase)

total_batch = data_pipeline.get_total_batch(train_xs, conf.batch_size)
data_pipeline.initialize_batch()
lr_decayed = conf.learning_rate * conf.decay_rate ** (global_step / total_batch)

total_vars = tf.trainable_variables()

e_vars = [var for var in total_vars if "gaussian" in var.name]
d_vars = [var for var in total_vars if "bernoulli" in var.name]

train_vars = e_vars + d_vars

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate = lr_decayed).minimize(total_loss, var_list = train_vars, global_step = global_step)

batch_v_xs, batch_vn_xs, batch_v_ys = data_pipeline.next_batch(valid_xs, valid_ys, conf.batch_size, make_noise= False)
plot_manifold_canvas(batch_v_xs[0:100], 10, "CelebA", name = "ori_image")
data_pipeline.initialize_batch()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

start_time = time.time()
for i in range(conf.n_epoch):
    loss_val = 0
    for j in range(total_batch):
        batch_xs,batch_noised_xs,_ = data_pipeline.next_batch(train_xs, train_ys, conf.batch_size, make_noise = False)
        feed_dict = {X:batch_xs, phase: True}
        l, _, lr_,g = sess.run([total_loss, optimizer, lr_decayed, global_step], feed_dict = feed_dict)
        loss_val += l/total_batch

    if i % 5 ==0 or i == (conf.n_epoch -1):
        images = sess.run(images_generated, feed_dict = {X: batch_xs,
                                                         phase: False})
        images = np.reshape(images, [-1, shape[1], shape[2], shape[3]])
        name = "Manifold_canvas_" + str(i)
        plot_manifold_canvas(images[0:100], 10, type = "CelebA", name = name)

    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr: %f    loss: %f    time: %d hour %d min %d sec" %(i, lr_, loss_val, hour, min, sec))

if conf.n_z == 2:
    test_total_batch = data_pipeline.get_total_batch(test_xs, conf.batch_size)
    data_pipeline.initialize_batch()
    latent_holder = []
    for i in range(test_total_batch):
        batch_test_xs, batch_test_noised_xs, batch_test_ys = data_pipeline.next_batch(test_xs,
                                                                                    test_ys,
                                                                                    conf.batch_size,
                                                                                    make_noise = False)
        feed_dict = {X: batch_test_xs,
                     phase: False}

        latent_vars = sess.run(z, feed_dict = feed_dict)
        latent_holder.append(latent_vars)
    latent_holder = np.concatenate(latent_holder, axis = 0)
    plot_2d_scatter(latent_holder[:,0], latent_holder[:,1], test_ys[:len(latent_holder)])
    print("complete plot_2d_scatter!")


