import pdb
import cv2
import numpy as np
from glob import *
import os
import sys
import random

random.seed(777)
np.random.seed(777)

class data_controller:
    def __init__(self, type, n_channel, normal, anomalus, num_normal_train, num_normal_test, num_abnormal_test,name):
        self.type = type
        self.n_channel = n_channel
        self.normal = normal
        self.anomalus = anomalus
        self.num_normal_train = num_normal_train
        self.num_normal_test = num_normal_test
        self.num_abnormal_test = num_abnormal_test
        self.name = name
        self.batch = 0
        self.debug = 0
        self.batch = 0
        self.enlr_size = 64
        self.name_img_n = []
        self.name_img_an = []
        self.n_images = []
        self.ab_images = []

        for i in self.normal:
            self.name_img_n = np.append(self.name_img_n, glob(os.path.join("./data/" + self.type, str(i) + "/*.*")))
        for j in self.anomalus:
            self.name_img_an = np.append(self.name_img_an, glob(os.path.join("./data/" + self.type, str(j) + "/*.*")))

        n_idx = np.random.permutation(len(self.name_img_n))[0:self.num_normal_train + self.num_normal_test]
        an_idx = np.random.permutation(len(self.name_img_an))[0:self.num_abnormal_test]

        if self.n_channel == 1:
            for k in n_idx:
                img = cv2.imread(self.name_img_n[k], flags = 0)
                img = cv2.resize(img,(self.enlr_size, self.enlr_size), interpolation=cv2.INTER_CUBIC)
                self.n_images.append(img)

            for a in an_idx:
                img = cv2.imread(self.name_img_an[a], flags = 0)
                img = cv2.resize(img, (self.enlr_size, self.enlr_size), interpolation=cv2.INTER_CUBIC)
                self.ab_images.append(img)

            self.n_images = np.reshape(self.n_images, [-1, self.enlr_size, self.enlr_size, self.n_channel])
            self.ab_images = np.reshape(self.ab_images, [-1, self.enlr_size, self.enlr_size, self.n_channel])
            self.test_images = np.append(self.n_images[self.num_normal_train:], self.ab_images,axis = 0)
            self.train_images = self.n_images[0:self.num_normal_train]
            np.random.shuffle(self.test_images)
        print("")
        print("-"*20, name, "-"*20)
        print("Train(normal) images: ", np.shape(self.train_images))
        print("Test(mixed) images: ", np.shape(self.test_images))

    def preprocessing(self):
        training_mean = np.mean(self.train_images, axis = (0,1,2))
        testing_mean = np.mean(self.test_images, axis = (0,1,2))
        self.train_images = self.train_images/255
        self.test_images = self.test_images/255
        print("Train_Data_mean: ", training_mean)
        print("Test_Data_mean: ", testing_mean)

        return self.train_images, self.test_images

    def initialize_batch(self):
        self.batch = 0

    def get_total_batch(self,images, batch_size):
        self.batch_size = batch_size

        return len(images)//self.batch_size

    def next_batch(self, ori_images, noised_images, batch_size):
        self.length = len(ori_images)//batch_size
        batch_xs = ori_images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
        batch_noised_xs = noised_images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
        self.batch += 1
        if self.batch == (self.length):
            self.batch = 0

        return batch_xs, batch_noised_xs


