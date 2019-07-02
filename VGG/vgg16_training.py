# -*- coding: utf-8 -*-
# 测试载入参数而非随机初始化 并且全连接层不载入预训练参数
# 某些层可以测试训练过程中是否参数变化
#%%

import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools
import parameter_load

#%%
IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000   
IS_PRETRAIN = True


#%%   Training
def train():
    
    pre_trained_weights = './vgg16_pretrain/vgg16.npy'
    data_dir = './data/cifar-10-batches-bin/'
    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'
    
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=True,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True)
        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=False,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False)
    
    logits = VGG.VGG16N(tra_image_batch, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, tra_label_batch)
    accuracy = tools.accuracy(logits, tra_label_batch)
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    # 占位符控制送入的是train_batch还是val_batch
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])    
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   
       
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # 载入预训练的参数，并且不载入3个全连接层
    # 如果这一行注释掉 则参数都是随机初始化
    parameter_load.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

