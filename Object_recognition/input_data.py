# -*- coding: utf-8 -*-

# input.py: 导入数据并生成训练批次
import tensorflow as tf
import numpy as np
import os
import math


#%% 超参数
trainImg_dir = './data/train/'  # 训练数据的相对路径
radio = 0.05
image_w = 224
image_h = 224
batch_size = 64
capcity = 256

#%% 
# 获取训练数据文件夹下所有文件路径，并赋予相应得标签
# 返回随机打乱后的训练图片文件路径 及其标签
'''
Args:
    file_dir: 训练图片数据的路径
    radio: 训练数据和验证数据的比列
Return:
    list of images and labels
'''
def get_file(file_dir, radio):
    wjj = []
    wjj_labels = []
    ztc = []
    ztc_labels = []
    for file in os.listdir(file_dir): #获取当前路径下得所有子路径（字符串）
        name = file.split(sep='.')  #根据文件名的特点分开得到各类数据路径 并打上标签
        if name[0] == 'wjj':
            wjj.append(file_dir + file)
            wjj_labels.append(0)
        else:
            ztc.append(file_dir + file)
            ztc_labels.append(1)
    print('There are %d wjj and %d ztc' % (len(wjj), len(ztc)))
    
    image_list = np.hstack((wjj, ztc)) #路径放在一起 ['a','b'] ['c','d']->['a','b','c','d']
    label_list = np.hstack((wjj_labels, ztc_labels)) #[1,1] [0,0] -> [1,1,0,0]
    tmp = np.array([image_list, label_list])    # [['a','b','c','d'], [1,1,0,0]]
    tmp = tmp.transpose()   #[['a',1], ['b',1], ['c',0], ['d',0]]
    np.random.shuffle(tmp)  #随机打乱 不按读取顺序生成batch，上面四步是为了np里这样操作很简单
    
    all_image_list = tmp[:,0]  # ['c', 'a', 'b', 'd']
    all_label_list = tmp[:,1]  # [0, 1, 1, 0]
    
    n_sample = len(all_image_list)
    n_val = math.ceil(n_sample*radio)  #radio计算训练数据中多少用来验证
    n_train = n_sample-n_val
    
    train_imgs = all_image_list[0 : n_train]  # 训练图片路径list
    train_labels = all_label_list[1 : n_train]  #相应的标签list
    train_labels = [int(float(i)) for i in train_labels]
    
    val_imgs = all_image_list[n_train : -1]  # 验证图片list
    val_labels = all_label_list[n_train : -1]   # #相应的标签list
    val_labels = [int(float(i)) for i in val_labels] 
    
    return train_imgs, train_labels, val_imgs, val_labels

#%%
# 为训练图片生成batch
'''
Args:
    image: get_file函数返回的train_imgs (list)
    label: get_file函数返回的train_labels (list)
    image_w,image_h: 图片归一化为统一的大小
    batch_size: batch的大小
    capacity: 队列容量
Return:
    image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size], dtype=tf.int32
'''
def get_batch(image, label, image_w, image_h, batch_size, capacity):
    image = tf.cast(image, tf.string) # tf.cast把python的数据类型转换成tensorflow识别的类型
    label = tf.cast(label, tf.int32)
    # 定义一个tensorflow输入的队列，这里image和label是分开的，用slice_input_producer
    # image和label组成一个list成为第一个参数
    input_queue = tf.train.slice_input_producer([image, label])
    # 队列里第二个位置是label，直接读取
    label = input_queue[1]
    # 图片需要经过一系列操作才能得到图片真正的数据
    image_contents = tf.read_file(input_queue[0]) # read
    image = tf.image.decode_jpeg(image_contents, channels=3) # decode
    ######################################
    # data argumentation should go to here
    ######################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h) # resize
    #image = tf.image.per_image_standardization(image) # 标准化
    
    # tf.train.batch根据队列里的数据生成批次
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

#%%
    
    