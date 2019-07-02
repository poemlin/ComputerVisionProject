# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 看前20张训练图片的label
for i in range(20):
    # 得到one-hot表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    one_hot_label = mnist.train.labels[i,:]
    # 通过np.argmax我们可以直接获得原始的label
    label = np.argmax(one_hot_label)
    
    print("mnist_train_%d label: %d" % (i, label))