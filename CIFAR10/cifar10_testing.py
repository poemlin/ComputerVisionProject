# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import numpy as np
import math
import cifar10_input
import cifar10_model

#%%
BATCH_SIZE = 64
N_CLASSES = 10
#%%

def evaluate():
    with tf.Graph().as_default():
        
        log_dir = './logs/train/'
        test_dir = './data/cifar-10-batches-bin/'
        n_test = 10000
        
        
        # reading test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits = cifar10_model.inference(images, BATCH_SIZE, N_CLASSES)
        # 比较真实label和预测值
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            # 读取模型文件
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE)) # 10000/64 取整
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE # 我们测试的数量
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
    
#%%