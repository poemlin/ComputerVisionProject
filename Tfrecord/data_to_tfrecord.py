# -*- coding: utf-8 -*-

'''
Transform data to tfrecord
'''
#%%

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io


#%%
# 获得所有图片路径 并赋予相应的label
'''
Args:
    file_dir: file directory
Returns:
    images: image directories, list, string
    labels: label, list, int
'''
def get_file(file_dir):
    images = []
    labels = []
    folders = []
    
    for root, sub_folders, file in os.walk(file_dir):
        for name in file:
            images.append(os.path.join(root, name)) # 所有图片
        for name in sub_folders:
            folders.append(os.path.join(root, name)) # 所有子目录
    
    for one_folder in folders:
        n_img = len(os.listdir(one_folder)) # 子文件夹下图片数量
        label_letter = one_folder.split('/')[-1] #子文件夹名，以此分label
        # label list依次加入label
        if label_letter=='A':
            labels = np.append(labels, n_img*[1])
        elif label_letter=='B':
            labels = np.append(labels, n_img*[2])
        elif label_letter=='C':
            labels = np.append(labels, n_img*[3])
        elif label_letter=='D':
            labels = np.append(labels, n_img*[4])
        elif label_letter=='E':
            labels = np.append(labels, n_img*[5])
        elif label_letter=='F':
            labels = np.append(labels, n_img*[6])
        elif label_letter=='G':
            labels = np.append(labels, n_img*[7])
        elif label_letter=='H':
            labels = np.append(labels, n_img*[8])
        elif label_letter=='I':
            labels = np.append(labels, n_img*[9])
        else:
            labels = np.append(labels, n_img*[10])
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
             
    return image_list, label_list

#%% copy from tf.org
# label转换为int64 image转换为byteslist
def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%% 把image和label转换成tfrecord
'''
Args:
    images: list of image directories, string type
    labels: list of labels, int type
    save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
    name: the name of tfrecord file, string type, e.g.: 'train'
Return:
    no return
Note:
        converting needs some time, be patient...
'''
def convert_to_tfrecord(images, labels, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    
    # 比较图片和label维度是否一致
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' 
                         %(images.shape[0], n_samples))    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename) # 重点函数python_io.TFRecordWriter
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i]) # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label':int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
#%%






    











