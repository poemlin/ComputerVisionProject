# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)   #读取mnist数据集

x = tf.placeholder(tf.float32, [None, 784])  # x是一个占位符，代表待识别的图片
y_ = tf.placeholder(tf.float32, [None, 10])  # y_是实际的图像标签，同样以占位符表示。

w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b) # softmax就是logit回归的多分类模型

'''
输入784维向量 输出10维向量，全连接参数就是784*10

加入输入一个数据，x = 1*784, w = 784*10
x = (1, 2, 3, 4, ..., 783, 784)

w = (1, 2, 3, 4 ,5, 6, 7, 8, 9, 10
     1, 2, 3, 4 ,5, 6, 7, 8, 9, 10
     1, 2, 3, 4 ,5, 6, 7, 8, 9, 10      784行10列
     ......
     1, 2, 3, 4 ,5, 6, 7, 8, 9, 10
     1, 2, 3, 4 ,5, 6, 7, 8, 9, 10)

mutmul(x, w) = 1*10 即y tensor的shape
'''

# 模型输出y，实际标签y_,shape都是[none,10]
# 我们希望两者越相似越好，交叉熵损失就可以形容这种相似性
# 损失越小，模型的输出就和实际标签越接近，模型的预测也就越准确 。
# -y_*logy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# 随机梯度下降优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession() # 创建一个Session
tf.global_variables_initializer().run() # 运行之前必须要初始化所有变量，分配内存。
print("start training...")

# 进行1000步梯度下降
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})


# 测试
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1)) # [batchsize(none)]大小tensor

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #True->1,求平均

# 由于x和y_size都是none 所以可以直接放入
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #0.9188














