# -*- coding: utf-8 -*-

# 对input里的两个函数进行测试
import matplotlib.pyplot as plt
import input_data
import tensorflow as tf
import numpy as np


BATCH_SIZE = 3
CAPCITY = 256
IMG_W = 208
IMG_H = 208
RADIO = 0.1
TRAIN_DIR = './data/train/'

# get_file获得图片路径和相应的标签
train_images, train_labels, val_images, val_labels = input_data.get_file(TRAIN_DIR,
                                                                    RADIO)

# get_batch生成batch                                                                 
train_image_batch, train_label_batch = input_data.get_batch(train_images,
                                                       train_labels,
                                                       IMG_W,
                                                       IMG_H,
                                                       BATCH_SIZE,
                                                       CAPCITY)
# 前面所有操作都只是一个空壳子，使用tensorflow的操作必须启动的session才能真正执行

with tf.Session() as sess:
    i = 0
    # 我们需要启动队列生成batch，需要这两个函数控制入列和出列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i<1:
            # 执行前面生成batch的真正操作 sess.run(前面定义的操作)
            img, label = sess.run([train_image_batch, train_label_batch])
            
            # 对一个batch的图片显示
            for j in np.arange(BATCH_SIZE):
                # 显示label
                print('label: %d' % label[j])
                # 四Dtensor，显示图片
                plt.imshow(img[j,:,:,:])
                plt.show()
                i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
    
