# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading


import numpy as np
import tensorflow as tf
import logging
#%%
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # PNG->JPG.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # decodes JPEG 
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
#%%
def _is_png(filename):
    """判断一张图片是否是PNG格式
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename # string匹配

#%%
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%
# tfrecord的feature都要写入example里，再存为tfrecords文件
def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example
#%%
def _process_image(filename, coder):
    """处理单张图片， 图片地址->解码后的结果
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # 读取图片文件
    with open(filename, 'rb') as f:
        image_data = f.read()

    # 如果是png，先转换为jpg
    if _is_png(filename):
        logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # 解码jpg图片
    image = coder.decode_jpeg(image_data)

    # 确保图片是被解码成rgb了
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


#%%
# 处理分配的一个批次的文件
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, command_args):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%s_%.5d-of-%.5d.tfrecord' % (command_args.dataset_name, name, shard, num_shards)
        output_file = os.path.join(command_args.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                logging.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                             (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        logging.info('%s [thread %d]: Wrote %d images to %s' %
                     (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    logging.info('%s [thread %d]: Wrote %d images to %d shards.' %
                 (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()
#%%
# 处理图片并保存为tfrecord
'''
Args:
      name: string, 数据集的标识
      filenames: list of strings; each string '/d/data/train/dog/1.jpg'
      texts: list of strings; each string 'dog'
      labels: list of integer; each integer 0 1 2
      num_shards: 处理这个数据集的线程数量
'''
def _process_image_files(name, filenames, texts, labels, num_shards, command_args):
    # 断言 三个参数大小相同
    assert len(filenames) == len(texts)
    assert len(texts) == len(labels)
    
    # 根据线程数量等量划分数据集
    # len(filenames)=10, num_threads=5
    spacing = np.linspace(0, len(filenames), command_args.num_threads + 1).astype(np.int) # [ 0  2  4  6  8 10]
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]]) # [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]] 数据集等线程数划分batch
    # 
    logging.info('Launching %d threads for spacings: %s' % (command_args.num_threads, ranges))
    sys.stdout.flush()
    # 协调器监视所有线程完成
    coord = tf.train.Coordinator()
    
    # coder类用来将图片解码成tf可以使用的格式
    coder = ImageCoder()
    
    threads = []
    # 分进程处理
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards, command_args)
        # 处理分配的数据 加入进程队列。[0,2]->threading
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    logging.info('%s: Finished writing all %d images in data set.' %
                 (datetime.now(), len(filenames)))
    sys.stdout.flush()

#%%
# 为数据集里的图片文件和标签建立一个list
'''
Args:
    data_dir: string, 图片数据的路径，路径格式为：data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg.'dog'是这些图片的标签.
    labels_file: string, 标签文件的路径，格式是：
          dog
          cat
          flower，每一行代表一个类别标签
Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
'''
def _find_image_files(data_dir, labels_file, command_args):
    logging.info('Determining list of input files and labels from %s.' % data_dir)
    # unique_labels = [dog, cat, ..., flower]
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()] # 从label文件获取label实际名的list
    
    labels = []
    filenames = []
    texts = []
    
    label_index = command_args.class_label_base # 从命令参数里获取标签的起始数字，一般为0  
    
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text) # '/home/data/train/dog/*' 某类所有图片目录
        matching_files = tf.gfile.Glob(jpeg_file_path) # 获取该目录下所有图片路径，返回list [/home/data/train/dog/1.jpg,/home/data/train/dog/2.jpg ...]
        labels.extend([label_index] * len(matching_files)) # [0] -> [0,0,0,0...,0]
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)
        
        if not label_index % 100: # 建立完一类（一个目录）记录日志，这里默认类别不超过100
            logging.info('Finished finding files in %d of %d classes.' % (label_index, len(labels)))
        
        label_index += 1
    
    # 前面获取得都是有序的图片和标签，这里打乱他们的顺序
    # 不加list会报错TypeError: 'range' object does not support item assignment
    shuffled_index = list(range(len(filenames))) # n=100, [0,1,2,3...,99]
    random.seed(12345) # 设置随机种子，使下面三次的随机结果相同
    random.shuffle(shuffled_index) # 随机打乱[0,1,..,99]
    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    
    logging.info('Found %d JPEG files across %d labels inside %s.' %
                 (len(filenames), len(unique_labels), data_dir)) # 获取完毕写入log
    
    return filenames, texts, labels

#%%
def _process_dataset(name, directory, num_shards, labels_file, command_args):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    filenames, texts, labels = _find_image_files(directory, labels_file, command_args)
    _process_image_files(name, filenames, texts, labels, num_shards, command_args)

#%%
# 检查并设置默认的命令行参数 
def check_and_set_default_args(command_args):
    # 如果传入的参数没有该属性，或者该属性在参数值值为None，则为它设置默认的参数
    if not(hasattr(command_args, 'train_shards')) or command_args.train_shards is None:
        command_args.train_shards = 5
    if not(hasattr(command_args, 'validation_shards')) or command_args.validation_shards is None:
        command_args.validation_shards = 5
    if not(hasattr(command_args, 'num_threads')) or command_args.num_threads is None:
        command_args.num_threads = 5
    if not(hasattr(command_args, 'class_label_base')) or command_args.class_label_base is None:
        command_args.class_label_base = 0
    if not(hasattr(command_args, 'dataset_name')) or command_args.dataset_name is None: # label起始默认为0
        command_args.dataset_name = 0
    # 运行程序时 下列参数不能省略，利用assertIsNotNone断言
    # 如果这些断言被违反了，会直接引起一个简单而又直接的失败。
    assert command_args.train_directory is not None
    assert command_args.validation_directory is not None
    assert command_args.labels_file is not None
    assert command_args.output_directory is not None
#%%
# 主函数，获取命令行参数，运行程序
def main(command_args):
    '''
    command_args:需要有以下属性：
    command_args.train_directory  训练集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.validation_directory 验证集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.labels_file 一个文件。每一行代表一个label名称。
    command_args.output_directory 一个文件夹，表示最后输出的位置。

    command_args.train_shards 将训练集分成多少份。
    command_args.validation_shards 将验证集分成多少份。
    command_args.num_threads 线程数。必须是上面两个参数的约数。

    command_args.class_label_base 很重要！真正的tfrecord中，每个class的label号从多少开始，默认为0（在models/slim中就是从0开始的）
    command_args.dataset_name 字符串，输出的时候的前缀。

    图片不可以有损坏。否则会导致线程提前退出。
    '''
    check_and_set_default_args(command_args)
    logging.info('Saving results to %s' % command_args.output_directory)
    
    # run
    _process_dataset('validation', command_args.validation_directory,
                     command_args.validation_shards, command_args.labels_file, command_args)
    _process_dataset('train', command_args.train_directory,
                     command_args.train_shards, command_args.labels_file, command_args)
    
#%%




































































