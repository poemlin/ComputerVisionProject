# -*- coding: utf-8 -*-
#%% 导入基本的包
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pylab

#%%
# python解析器会搜索当前目录、已安装的内置模块和第三方模块，搜索路径存放在sys模块的path
# sys.path 返回的是一个列表,当我们要添加自己的搜索目录时，可以通过列表的append()方法；
# 将上层目录导入进来，这样才可以执行这下面的两条导入命令
sys.path.append("..")
# 从detection modules里导入的包
from utils import label_map_util
from utils import visualization_utils as vis_util

#%% 模型准备与定义（定义使用的模型名，地址，以及label的位置
# 选择一个官方训练好的模型，这里是使用ssd+mobilenet在coco数据集上训练好的模型
# 在g3doc文件夹里，打开detection_model_zoo可以选择自己想要的模型
# ssd_inception、rfcn_resnet101、faster_rcnn_resnet101等等
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

MODEL_FILE = MODEL_NAME + '.tar.gz' #下载好的文件压缩包名称
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/' # 官方训练好的模型主地址

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' # 最终解压后的pb文件地址 model_name/frozen_inference_graph.pb

# index到类名的映射，这里用的是coco数据集，所以使用coco的映射，该文件在data文件夹内
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90 # coco一共90类检测物体

#%% 模型下载
# opener = urllib.request.URLopener() # 请求url
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE) #把url的东西保存到本地
# tar_file = tarfile.open(MODEL_FILE) # 打开压缩包
# 压缩包内有多个文件，但是我们只需要pb文件（同时包含网络结构和参数）作为测试使用
# 所以我们需要选择性解压缩
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())
# 最终同级目录存在MODEL_FILE压缩包文件以及MODEL_NAME文件夹，里面存放pb文件
#%% 将pb文件读取到默认的计算图中，并且存放到内存内
detection_graph = tf.Graph() # 新建计算图

with detection_graph.as_default(): # 新建的计算图作为默认图
    od_graph_def = tf.GraphDef() # 重定义计算图
    with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid: # 读取pb文件
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph) # 把读取的pb文件放到计算图中
        tf.import_graph_def(od_graph_def, name='') # 导入建立并且已经读取的计算图到内存

#%% 建立从index到类名的转换，以便显示。
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#%% 将图片转换为numpy数组格式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

#%% 测试图片的路径
PATH_TO_TEST_IMAGES_DIR = 'test_images' # 同级目录下的test_images文件夹
# LIST, 测试图片：[image1.jpg image2.jpg image3.jpg]可自行设置
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]
IMAGE_SIZE = (12,8) # 输出图片的大小

#%% Detection session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess: # 前面detection_graph已经读取到内存中
    # 从计算流图中获取输入tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 从计算流图中获取计算图中的box，每个box代表一个检测到的物体
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 从计算流图中获取box的得分，代表模型给出的置信度
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # 从计算流图中获取box的类别，代表检测到物体的类别
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # 从计算流图中获取检测到物体的数量
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # image转换成numpy格式
      image_np = load_image_into_numpy_array(image)
      # 将三维numpy转换成四维tensor[1, None, None, 3]输入网络
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # sess.run真正计算
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # 可视化检测结果
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      pylab.show()

#%%





















