# -*- coding: utf-8 -*-
import data_to_tfrecord
import read_tfrecord
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#%%
test_dir = './data/'
save_dir = './result/'
BATCH_SIZE = 25
name_test = 'test'

#%%
images, labels = data_to_tfrecord.get_file(test_dir) # 获取image和label
data_to_tfrecord.convert_to_tfrecord(images, labels, save_dir, name_test) #转换至tfrecord


#%% TO test train.tfrecord file
# 显示一个batch的图片 5*5=25
def plot_images(images, labels):
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[i] - 1), fontsize = 14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

#%%
tfrecords_file = './result/test.tfrecords'
img_w = 28
img_h = 28
image_batch, label_batch = read_tfrecord.read_and_decode(tfrecords_file, batch_size=BATCH_SIZE,
                                                         img_w, img_h)

with tf.Session()  as sess:
    
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            # just plot one batch size            
            image, label = sess.run([image_batch, label_batch])
            plot_images(image, label)
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
    

#%%