# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cifar10_input

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

#%%
data_dir = '/data/minglin/data/cifar10_data/cifar-10-batches-bin/'
batch_size = 4

image_batch, label_batch = cifar10_input.read_cifar10(data_dir,
                                                      is_train=True,
                                                      batch_size=batch_size,
                                                      shuffle = True)

with tf.Session() as sess:
    i = 0 # 控制只运行一个batch
    coord = tf.train.Coordinator() # 开启协调器
    threads = tf.train.start_queue_runners(coord=coord) # 开启队列
    
    try:
        while not coord.should_stop() and i<1:
            images, labels = sess.run([image_batch, label_batch])
            # 显示label和image
            for j in np.arange(batch_size):
                print('labels: %d' % np.argmax(labels[j]))
                plt.imshow(images[j,:,:,:])
                plt.show()
                
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
    
    


#%%
