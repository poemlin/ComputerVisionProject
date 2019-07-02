# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import os
import cifar10_input
import cifar10_model
import numpy as np
import os.path

#%%
BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 10000 # 时间大概不到半小时
N_CLASSES = 10
#%% Train the model on the training data
# you need to change the training data directory below

def train():
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    
    
    data_dir = './data/cifar-10-batches-bin/'
    log_dir = './logs/train/'
    
    images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size= BATCH_SIZE,
                                                shuffle=True)
    logits = cifar10_model.inference(images, BATCH_SIZE, n_classes=N_CLASSES)
    
    loss = cifar10_model.losses(logits, labels)
    
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) # 定义优化器
    train_op = optimizer.minimize(loss, global_step= my_global_step) # 运行优化
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, loss_value = sess.run([train_op, loss])
               
            if step % 50 == 0:                 
                print ('Step: %d, loss: %.4f' % (step, loss_value))
                
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)                
    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
#%%