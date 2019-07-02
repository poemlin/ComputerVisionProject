# -*- coding: utf-8 -*-

import tensorflow as tf
#%%
# 定义了整个网络的结构
# 网络结构和官网cifar10教程中网络结构一样c-p-c-p-fc-fc-softmax
'''
Args:
    images: image batch, tf.float32, 4Dtensor [batch_size, w, h, channel]
Return:
    tensor with computed logits, float: [batch_size, n_classes]
'''
def inference(images, batch_size, n_classes):
    # 卷积过程包含：卷积shape[kernel_size, kernel_size, channel, kernel_number]卷积strides
    # variable_scope会使tensorboard形成node
    # conv1
    with tf.variable_scope('conv1') as scope:
        # 定义weights和biases
        weights = tf.get_variable('weights',
                                  shape = [3, 3, 3, 16],
                                  dtype = tf.float32,
                                  # 初始化方式选择和初始值设置对训练都至关重要
                                  initializer = tf.truncated_normal_initializer(
                                          stddev = 0.1, dtype = tf.float32)) #均值默认0
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        
        conv = tf.nn.conv2d(images, weights, strides = [1,1,1,1], padding='SAME')
        add_bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(add_bias, name=scope.name)
    # pool1
    with tf.variable_scope('pool1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                               padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3, 3, 16, 16],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(
                                          stddev = 0.1, dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding='SAME')
        add_bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(add_bias, name = scope.name)
    # pool2
    with tf.variable_scope('pool2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pool1')
    # fc3
    with tf.variable_scope('fc3') as scope:
        # 拉平以便进入全连接， -1表示未知，自己计算
        flat = tf.reshape(pool2, shape=[batch_size, -1])
        dim = flat.get_shape()[1].value
        
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                          stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(flat, weights) + biases, name = scope.name)
    # fc4   
    with tf.variable_scope('fc4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                          stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc4 = tf.nn.relu(tf.matmul(fc3, weights)+biases, name = scope.name)
    # softmax
    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                          stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 注意这里没有使用激活函数，后面的loss会计算
        softmax_linear = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)
    
    return softmax_linear
        

        
# %%
# 计算模型的损失
'''
Args:
    logits: inference的返回值，tensor,float,[batch_size, n_classes]
    labels: label tensor, int, [batch_size]
return:
    loss tensor float type
'''
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        # 包含sparse的softmaxloss不需要标签是onehot
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
# 优化loss的训练
'''
Args:
    loss: loss tensor return from losses
return:
    train_op: the op for training
'''
def training(loss, learning_rate):
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) # 选择优化算法
        # 就像任何Tensor（张量）一样，使用Variable（）创建的变量可以用作图中其他操作的输入
        global_step = tf.Variable(0, name = 'global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step) # 运行优化算法
    return train_op

# %%
# 计算误差准确率
'''
Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size]
returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
'''
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        # 每个logits最大值处的索引与label比较，相同返回True
        # logits = tf.nn.softmax(logits) #此处计算topk为何不加softmax？softmax只是归一化不影响？
        correct = tf.nn.in_top_k(logits, labels, 1)
        # bool->float
        correct = tf.cast(correct, tf.float16)
        # reduce_mean正好可以计算正确率
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
# %%
        