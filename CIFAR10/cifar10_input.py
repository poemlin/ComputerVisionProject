# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import numpy as np
import os

#%% 读取二进制格式的cifar10 并产生image batch和label batch
'''
Args:
    data_dir: the directory of CIFAR10
    is_train: boolen, 是否是训练数据
    batch_size:
    shuffle: 是否需要打乱
Returns:
    label: 1D tensor, tf.int32
    image: 4D tensor, [batch_size, height, width, 3], tf.float32
'''
def read_cifar10(data_dir, is_train, batch_size, shuffle):
    image_w = 32
    image_h = 32
    image_depth = 3
    label_bytes = 1
    image_bytes = image_w*image_h*image_depth
    
    with tf.name_scope('input'):
        if is_train: # 加载训练二进制数据
            filename = [os.path.join(data_dir, 'data_batch_%d.bin' %ii) 
                                        for ii in np.arange(1, 6)]
        else:   # 加载测试二进制数据
            filename = [os.path.join(data_dir, 'test_batch.bin')]
        # 产生一个tf的数据w文件名输入队列（这里image和label在一起 用string_input
        input_queue = tf.train.string_input_producer(filename)
        # 定义固定长度的recoderreade 32*32*3+1=3073代表一张图片
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes) 
        key, value = reader.read(input_queue)
        record_bytes = tf.decode_raw(value, tf.uint8) # 解码value信息
        
        label = tf.slice(record_bytes, [0], [label_bytes]) # 切出解码信息的第一个元素label
        label = tf.cast(label, tf.int32)
        # 切出剩余信息 仍是向量 需要reshape
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [image_depth, image_h, image_w])
        image_raw = tf.transpose(image_raw, (1,2,0)) # D/H/W to H/W/D
        image = tf.cast(image_raw, tf.float32)
        
#        # 此处数据增广操作
#        image = tf.random_crop(image, [24,24,3]) # 随机裁剪
#        image = tf.image.random_flip_left_right(image) # 随机反转
#        image = tf.image.random_brightness(image, max_delta=63) # 随机亮度
#        image = tf.image.random_contrast(image, lower=0.2, upper=1.8) # 随机对比度
        
        # image = tf.image.per_image_standardization(image) # 减去均值除以方差
        
        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=16,
                                                      capacity=2000,
                                                      min_after_dequeue=1500)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=16,
                                                      capacity=2000)
            
#        # label不用onehot
#        return image_batch, tf.reshape(label_batch, [batch_size])
    
        # label使用onehot
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        
        return image_batch, tf.reshape(label_batch, [batch_size, n_classes])

#%%