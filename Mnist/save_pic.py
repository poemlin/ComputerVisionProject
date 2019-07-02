# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 读取MNIST数据集。如果不存在会事先下载。
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 生成的图片存放在MNIST_data/raw/文件夹下
save_dir = "MNIST_data/raw/"

# 如果没有这个文件夹会自动创建
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    
# 保存前2张图片
for i in range(2):
    # 获取第i张图片的向量表示
    image_array = mnist.train.images[i,:]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    file_name = save_dir + 'mnist_train_%d.jpg' % i # 文件名称
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(file_name)

print('please check: %s' % save_dir)