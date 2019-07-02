# -*- coding: utf-8 -*-
# 使用imagenet训练好的参数 是很重要的 也是迁移学习的关键
# .npy 是python的一种数据格式
import tensorflow as tf
import numpy as np

#%% 载入
def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()
    
    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))
                
                

#%% 测试载入参数的shape
                
def test_load():
    data_path = './vgg16_pretrain/vgg16.npy'
    
    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)  

              
#%% 跳过某些层 不载入 这些层使用随机初始化      
# 注意 skip和trainable的区别 一个是否载入参数用于迁移，一个是该层训练过程中参数是否变化
def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))



























