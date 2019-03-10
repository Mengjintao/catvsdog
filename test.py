# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:55:16 2019

@author: yuanjianrui
@description: 测试训练的结果
"""

import tensorflow as tf 
import numpy as np 
import pdb
from datetime import datetime
from VGG16 import *
from PIL import Image
import cv2
import os
import VGG16
 




#=======================================================================
#获取一张图片
def get_one_image(train):
    #输入参数：train,训练图片的路径
    #返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   #随机选择测试的图片
    print(img_dir)
    img = Image.open(img_dir)
    #plt.imshow(img)
    imag = img.resize([64, 64])  #由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image






def test(path):
 
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, 17)
    score = tf.nn.softmax(output)
    f_cls = tf.argmax(score, 1)
 
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 训练好的模型位置
    saver.restore(sess, './model/model.ckpt-4999')
    for i in os.listdir(path):
        imgpath = os.path.join(path, i)
        im = cv2.imread(imgpath)
        im = cv2.resize(im, (224 , 224))# * (1. / 255)
 
        im = np.expand_dims(im, axis=0)
        # 测试时，keep_prob设置为1.0
        pred, _score = sess.run([f_cls, score], feed_dict={x:im, keep_prob:1.0})
        prob = round(np.max(_score), 4)
        print ("{} direction class is: {}, score: {}".format(i, int(pred), prob))
 
        
    sess.close()
 
 
if __name__ == '__main__':
    # 测试图片保存在文件夹中了，图片前面数字为所属类别
    path = 'D:\\dst\\data'
    test(path)