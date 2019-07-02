# -*- coding: utf-8 -*-

# 定义了很多常用的接口，这些接口可以使模型跟价简单，并且这些接口可以直接复用

#%%
import tensorflow as tf
import numpy as np

#%% 卷积操作接口，包含激活函数
'''
Args:
    layer_name: e.g. conv1, pool1...
    x: input tensor, [batch_size, height, width, channels]
    out_channels: number of output channels (or comvolutional kernels)
    kernel_size: the size of convolutional kernel
    stride: A list of ints. 1-D of length 4.
    is_pretrain:
    # trainabel是get_variable()里的参数，只要为false，参数就是固定住的不会改变。
    是true就是可以训练改变的
    the parameters of freezed layers will not change when training.
Returns:
    4D tensor， [batch_size, height, width, channels]
    '''
def conv(layer_name,x,out_channels,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=True):
    in_channels = x.get_shape()[-1] # 计算输入feature的in_channel
    with tf.variable_scope(layer_name):
        # get_variable是tf.org推荐的参数初始化方式，产生的参数可以重复使用
        # get_variable同时可以使用已经训练好的参数赋值
        w = tf.get_variable(name = 'weights',
                            trainable = is_pretrain,
                            shape=[kernel_size[0],kernel_size[1],in_channels,out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) #x初始化器更加科学
        b = tf.get_variable(name = 'biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu') # 包含激活函数
        
        return x

#%% 池化操作
'''
Args:
    x: input tensor [batch_size, height, width, channels]
    kernel: pooling kernel
    stride: stride size
    padding:
    is_max_pool: boolen 最大卷积还是均值卷积
'''
def pool(layer_name, x, kernel=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, stride, padding='SAME', name=layer_name)
    return x

#%% 批规范化操作 这里缩放scale和偏移offset没有设置
# [ ( x - mean ) / var ] * scale + offset
def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean = batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x

#%% 全连接层
'''
Args:
    layer_name: e.g. 'FC1', 'FC2'
    x: input feature map
    out_nodes: number of neurons for current FC layer
'''
def fc_layer(layer_name, x, out_nodes):
    shape = x.get_shape()
    if len(shape)==4: # 如果第一次进入 [batch_size,w,h,c]->[batch_size, w*h*c]
        in_nodes = shape[1].value*shape[2].value*shape[3].value
    else: # [batch_size, in_nodes]
        in_nodes = shape[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[in_nodes, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        
        flat_x = tf.reshape(x, [-1, in_nodes])
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        
        return x
    
#%% 计算模型损失
'''
Args:
    logits: logits tensor, [batch_size, n_classes]
    labels: labels
    is_onehot: 是不是onehot label
'''
def loss(logits, labels, is_onehot=True):
    with tf.variable_scope('loss') as scope:
        
        labels = tf.cast(labels, tf.int64)
        
        if is_onehot: # 如果label是onehot
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                            (logits=logits, labels=labels, name='xentropy_per_example')
        else: # 如果label不是onehot               
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                            (logits=logits, labels=labels, name='xentropy_per_example')
                        
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss

#%% 计算正确率
'''
Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
'''
def accuracy(logits, labels):

  with tf.name_scope('accuracy') as scope:
      # 这里label是onehot，不是，使用in_top_k
      # logit最大值的索引（预测类别）和实际label最大值的索引（真实类别）是否相等
      correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy

#%% 计算计算正确的个数
def num_correct_prediction(logits, labels):
    with tf.name_scope('num_correct'):    
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.int32)
        n_correct = tf.reduce_sum(correct)
    return n_correct

#%% 定义优化器并优化
def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

#%% 打印参数
'''
train_only : boolean
    If True, only print the trainable variables, otherwise, print all variables.
'''
def print_all_variables(train_only=True):
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   

#%%  

















