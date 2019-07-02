# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%
N_CLASSES = 2
IMG_W = 208
IMG_H = 208
RADIO = 0.2 # take 20% of dataset as validation data 
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 8000
LEARNING_RATE = 0.0001 # begain train suggested to use learning rate<0.0001

#%%

def run_training():
    train_dir = './data/train/'
    logs_train_dir = './logs/train/'
    logs_val_dir = '/logs/val/'
    # 获得训练数据和label
    train, train_label, val, val_label = input_data.get_file(train_dir, RADIO)
    # 生成batch
    train_batch, train_label_batch = input_data.get_batch(train, train_label, 
                                                          IMG_W, IMG_H,
                                                          batch_size=BATCH_SIZE,
                                                          capacity=CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val, val_label,
                                                      IMG_W, IMG_H,
                                                      batch_size=BATCH_SIZE,
                                                      capacity=CAPACITY)
    
    logits = model.inference(images=train_batch,
                             batch_size=BATCH_SIZE,
                             n_classes=N_CLASSES)   #计算网络输出
    loss = model.losses(logits, train_label_batch) # 计算损失
    train_op = model.training(loss, LEARNING_RATE)  # 计算优化
    acc = model.evaluation(logits, train_label) # 计算模型输出和真实值的比较
    # 模型输入占位符
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_= tf.placeholder(tf.int16, shape = [BATCH_SIZE])
    
    with tf.Session() as sess:
        saver = tf.train.Saver() # 建立模型存储器
        sess.run(tf.global_variables_initializer()) # sess sun initializer
        coord = tf.train.Coordinator() # coord
        threads = tf.train.start_queue_runners(sess=sess, coord = coord) # strat queue
        
        summary_op = tf.summary.merge_all() # 合并所有参数进入tensorboard
        # sess.graph 写入logs
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
        
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                train_images, train_labels = sess.run([train_batch, 
                                                       train_label_batch])
                _, train_loss, train_acc = sess.run([train_op, loss, acc],
                                                    feed_dict={x:train_images,
                                                              y_:train_labels})
            # 每50个step计算一次训练loss
            if step % 50 == 0:
                print('Step %d, train loss = %.4f, train acc = %.4f%%' % (step, train_loss, 
                                                                          train_acc*100))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            # 每200个step计算一次验证集正确率
            if step % 200 == 0 or (step+1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc],
                                             feed_dict = {x:val_images, y_:val_labels})
                print('** Step %d, val loss = %.4f, val acc = %.4f%%' % (step, val_loss,
                                                                          val_acc*100))
                summary_str = sess.run(summary_op)
                val_writer.add_summary(summary_str, step)
            # 每2000个step保存一次模型
            if step % 2000 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt') # 新建文件
                saver.save(sess, checkpoint_path, global_step=step) # saver.save
        
        except tf.errors.OutOfRangeError:
            print('training done -- epoch limit')
        finally:
            coord.request_stop()
        coord.join(threads)
#%%            
            
                
    
    
    
    