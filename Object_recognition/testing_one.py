# -*- coding: utf-8 -*-
#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import input_data
import tensorflow as tf
import model
#%%
def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    # train是一个包含所有图片的list，随机选择一个
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    # 打开该图片 并显示
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image) #读取的图片变成nparray
    return image

#%%
def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    train_dir = './data/train/'
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3]) # reshpae成4Dtensor
        
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = './logs/train/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a wjj with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a ztc with possibility %.6f' %prediction[:, 1])
#%%
'''
测试所有测试图片的正确率，可以使用getfile getbatch（slice_input_producer里
numepoch设置为1） 以及model.evluation()
'''