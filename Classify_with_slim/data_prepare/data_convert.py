# coding:utf-8
from __future__ import absolute_import
import argparse
import os
import logging
from src.tfrecord import main
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorflow-data-dir', default='pic/')
    parser.add_argument('--train-shards', default=2, type=int)
    parser.add_argument('--validation-shards', default=2, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--dataset-name', default='satellite', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # 初始化日志 日志级别是info
    args = parse_args() # 获得命令行解析对象 用于上面的参数
    args.tensorflow_dir = args.tensorflow_data_dir # 命令行对象添加参数 'pic/'
    args.train_directory = os.path.join(args.tensorflow_dir, 'train') # 命令行对象添加参数 'pic/train/'
    args.validation_directory = os.path.join(args.tensorflow_dir, 'validation') #命令行对象添加参数 'pic/validation/'
    args.output_directory = args.tensorflow_dir # 命令行对象添加参数 tfrecord文件存放地方 'pic/'
    args.labels_file = os.path.join(args.tensorflow_dir, 'label.txt') # 添加参数,label名文件应该在 'pic/label.txt'(如果有)
    if os.path.exists(args.labels_file) is False: # 如果没有 根据文件名自动创建
        logging.warning('Can\'t find label.txt. Now create it.')
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')
    # run src.tfrecord.main
    main(args)
