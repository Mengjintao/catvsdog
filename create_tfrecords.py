# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:50:17 2019

@author: yuanjianrui
@description: 制作tfrecord格式数据，
"""

import os
import tensorflow as tf
from PIL import Image
import sys
import numpy as np
 
def creat_tf(imgpath):
 
    #cwd = os.getcwd()
    #classes = os.listdir(cwd + imgpath)
    
    # 此处定义tfrecords文件存放
    writer = tf.python_io.TFRecordWriter("/home/jtmeng/tmp/catvsdog/tfrecord")
    for name in os.listdir(imgpath):
        class_path = imgpath + name+'/'
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                print (img_path),
                img = Image.open(img_path)
                img = img.resize((224, 224))
                im_array = np.array(img)
                sz = im_array.shape
                if len(sz) == 3 and sz[2] == 3 :
#                    print (type(sz))
                    print (im_array.shape),
                    print (im_array.dtype)
                    img_raw = img.tobytes()        
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                    writer.write(example.SerializeToString()) 
#                print (img_name) 
    writer.close()
 
def read_example():
 
    #简单的读取例子：
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
    
        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(label)
 
if __name__ == '__main__':
    imgpath = '/home/jtmeng/tmp/catvsdog/Pet/'
    creat_tf(imgpath)
#    read_example()
