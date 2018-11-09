from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
import time

DEFINE_string("data", "MNIST", "[MNIST|MNIST_Fashion|CIFAR|CALTECH]")

DEFINE_integer("n_epoch", 100, "number of Epoch for training")
DEFINE_integer("n_z", 100, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 128, "Batch Size for training")

DEFINE_float("W_d_rate", 0.1, "Decay rate for weights regularization")
DEFINE_float("E_rate", 0.3, "Weight for Potential loss")
DEFINE_float("learning_rate", 0.00001, "learning_rate")

conf = print_user_flags(line_limit = 100)
print("-"*80)

FLAG_1_Data = data_controller(type = "MNIST",
                              n_channel = 1,
                              normal = [0,1,2,3,4,5,6,7,8,9],
                              anomalus = [0],
                              num_normal_train = 40000,
                              num_normal_test = 1800,
                              num_abnormal_test = 200,
                              name = "FLAG_1")

train_data,test_data = FLAG_1_Data.preprocessing()
noised_train_data = salt_pepper_noise(train_data)

plot_manifold_canvas(train_data[0:100],10,"MNIST", name ="original_images")
plot_manifold_canvas(noised_train_data[0:100],10,"MNIST", name ="noised_images")
plot_manifold_canvas(test_data[0:100],10, "MNIST", name = "test_images")

shape = np.shape(train_data)
W = shape[1]
H = shape[2]
C = shape[3]

X = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs")
X_noised = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs_noised")
phase = tf.placeholder(tf.bool, name = "training_phase")
global_step = tf.Variable(0, trainable = False, name = "global_step")

EBAnoGAN = EBAnoGAN(conf,shape, [128,256,512,1024])
X_out, N_log_likelihood = EBAnoGAN.EB_Variational_AutoEncoder(X,X_noised,phase)

total_batch = FLAG_1_Data.get_total_batch(train_data, conf.batch_size)
FLAG_1_Data.initialize_batch()

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer(learning_rate = conf.learning_rate).minimize(N_log_likelihood,
                                                                                    global_step = global_step)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

start_time = time.time()
for i in range(conf.n_epoch):
    loss_val = 0
    for j in range(total_batch):
        batch_xs, batch_noised_xs = FLAG_1_Data.next_batch(train_data, noised_train_data, conf.batch_size)
        feed_dict = {X: batch_xs, X_noised: batch_noised_xs, phase: True}
        l,_,g = sess.run([N_log_likelihood, optimizer, global_step], feed_dict = feed_dict)
        loss_val +=l/total_batch

    if i % 1 == 0 or i == (conf.n_epoch -1):
        images_plot = sess.run(X_out, feed_dict = {X_noised: batch_xs, phase: False})
        name = "Manifold_canvas_" + str(i)
        plot_manifold_canvas(images_plot[0:100],10, type = "MNIST", name = name)

    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr: %f    loss: %f    time: %d hour %d min %d sec\n" %(i, conf.learning_rate, loss_val, hour, min, sec))