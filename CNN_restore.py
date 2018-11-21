import glob
import os
import cv2;
import numpy as np;
import math
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import time
from sklearn.metrics import confusion_matrix
from resizeimage import resizeimage

def forward_propagation(X, W1, W2):
    co1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME',name="co1")
    A1 = tf.nn.relu(co1)
    po1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME',name="po1")
    co2 = tf.nn.conv2d(po1,W2, strides = [1,1,1,1], padding = 'SAME',name="co2")
    A2 = tf.nn.relu(co2)
    po2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME',name="po2")
    P2_f = tf.contrib.layers.flatten(po2)
    #fi = tf.contrib.layers.fully_connected(P2_f, 2, activation_fn=None
    fi = tf.layers.dense(inputs = P2_f,units=2, activation=None,name="dense",reuse = tf.AUTO_REUSE)
    return fi
###Resrtoration starts from here###
#Path where my file is stored#
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
new_width  = 480
new_height = 240
os.chdir('D:\\SUTD\\crackedglass\\newimage\\msav')
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph("my-model.meta")
    saver.restore(sess, "my-model")
    graph=tf.get_default_graph()    
    W1=graph.get_tensor_by_name("W1:0")
    W2=graph.get_tensor_by_name("W2:0")
    X = graph.get_tensor_by_name("Placeholder:0")
    Y = graph.get_tensor_by_name("Placeholder_1:0")
    #y_h1 = forward_propagation(X, W1, W2)
    y_h = tf.get_collection('y_h')[0]
    final = 0
    os.chdir('D:\\SUTD\\crackedglass\\newimage\\vid_te\\vid_te1')
    count_crack = 0
    count_uncrack = 0
    vidcap = cv2.VideoCapture('test.mp4');
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        image = None
        success,image = vidcap.read()
        print('read a new frame:',success)
        if success == True:
            image = cv2.resize(image, (480, 240))
       
        #time.sleep(1)
        
            count+=1
        #img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            img = rgb2gray(image)
            cv2.imwrite('frame%d.jpg'%count,img)
            img = np.reshape(img,(1,240,480,1))
            tr = img/255
            er2=sess.run([y_h], feed_dict={X: tr})
            fi = np.argmax(er2[0][0])
            if fi == 0:
                print("Cracked")
                count_crack = count_crack+1
            #ob_class.append(0)
            if fi == 1:
                print("Uncracked")
                count_uncrack = count_uncrack+1
            #ob_class.append(1)
