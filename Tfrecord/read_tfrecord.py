# -*- coding: utf-8 -*-
#%%
import tensorflow as tf

#%% 读取tfrecord文件 并且生成batch 可以直接送入网络训练
'''
Args:
    tfrecords_file: the directory of tfrecord file
    batch_size: number of images in each batch
Returns:
    image: 4D tensor - [batch_size, width, height, channel]
    label: 1D tensor - [batch_size]
'''
def read_and_decode(tfrecords_file, batch_size, img_w, img_h):
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    
    image = tf.reshape(image, [img_w, img_h])
    label = tf.cast(img_features['label'], tf.int32)    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = 2000)
    return image_batch, tf.reshape(label_batch, [batch_size])
#%%