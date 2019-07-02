# -*- coding: utf-8 -*-
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

# import tensorflow.contrib.slim as slim
slim = tf.contrib.slim

#%%
_FILE_PATTERN = 'satellite_%s_*.tfrecord' #tfrecord文件名
SPLITS_TO_SIZES = {'train':4800, 'validation':1200} #训练集和验证集的大小
_NUM_CLASSES = 6 # 数据集类别
_ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying size.',
        'label':'A single integer between 0 and 4',
        }

#%%
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  # 字典匹配查找key，根据train和validation关键字划分tfrecord文件
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern: #tfrecord文件名可以指定，如果没有，则默认是'satellite_%s_*.tfrecord'
    file_pattern = _FILE_PATTERN
  # #'satellite_train_*.tfrecord' 'satellite_validation_*.tfrecord'
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name) 

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      # 定义了图片的默认格式 。 收集的卫星图片的格式为 jpg 图片，因此修改为 jpg 。
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
#%%




















