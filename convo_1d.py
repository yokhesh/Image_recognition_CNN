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


np.random.seed(1)

###Cracked
con = []
os.chdir('D:\\SUTD\\crackedglass\\newimage\\cracked')
for file in glob.glob('*.jpg'):
        img = None
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img,(240,480,1))
        con.append(img)
cracked = np.asarray(con)
# Doubling the cracked image vectors
cracked = np.concatenate((cracked, cracked), axis=0)
#cracked training labels
ytrc = np.full((cracked.shape[0], 2), [1,0])
##Uncracked
os.chdir('D:\\SUTD\\crackedglass\\newimage\\uncracked')
con1 = []
for file in glob.glob('*.jpg'):
        
        img = None
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img,(240,480,1))
        con1.append(img)
uncracked = np.asarray(con1)

# Doubling the uncracked image vectors
uncracked = np.concatenate((uncracked, uncracked), axis=0)
#uncracked training labels
ytruc = np.full((uncracked.shape[0], 2), [0,1])
# Combining the cracked and uncracked features
train_vectors = np.concatenate((cracked, uncracked), axis=0)
#Normalize vectors
train_vectors = train_vectors/255
##training labels
train_labels = np.concatenate((ytrc, ytruc), axis=0)

##testing
os.chdir('D:\\SUTD\\crackedglass\\newimage\\testing')
con2 = []
for file in glob.glob('*.jpg'):
        img = None
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img,(240,480,1))
        con2.append(img)
testing = np.asarray(con2)
test_vectors = testing/255

##test labels
yte1 = np.full((4, 2), [1,0])
yte2 = np.full((3, 2), [0,1])
test_labels = np.concatenate((yte1, yte2), axis=0)
# Creating placeholders
def create_placeholders(r, s, t, q):
    X = tf.placeholder(tf.float32,[None, r, s, t])
    Y = tf.placeholder(tf.float32,[None, q])
    return X, Y
#Initialize parameters
def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    return W1, W2
#Forwrad prop
def forward_propagation(X, W1, W2):
    co1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(co1)
    print(A1)
    po1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    print(po1)
    co2 = tf.nn.conv2d(po1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(co2)
    print(A2)
    po2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    print(po2)
    P2_f = tf.contrib.layers.flatten(po2)
    print(P2_f)
    y_h = tf.contrib.layers.fully_connected(P2_f, 2, activation_fn=None)
    return y_h
def compute_cost(y_h, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_h, labels = Y))
    return cost
#Final model
os.chdir('D:\\SUTD\\crackedglass\\newimage\\msav')

def model(train_v, train_l,test_v, test_l,learning_rate = 0.009,
          num_epochs = 100,print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, r, s, t) = train_v.shape
    q = train_l.shape[1]
    costs = []
    X, Y = create_placeholders(r, s, t, q)
    W1, W2 = initialize_parameters()
    y_h = forward_propagation(X, W1, W2)
    cost = compute_cost(y_h, Y)
    optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    loss= []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        for epoch in range(num_epochs):
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: train_v, Y: train_l})
            print(temp_cost)
            loss.append(temp_cost)
        predict_op = tf.argmax(y_h, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: train_v, Y: train_l})
        test_accuracy = accuracy.eval({X: test_v, Y: test_l})
        print(test_accuracy)
        ## Testing
        
        er2=sess.run([y_h], feed_dict={X: test_v})
        tf.add_to_collection('y_h', y_h)
        saver.save(sess, './my-model')
        os.chdir('D:\\SUTD\\crackedglass\\newimage')
        print("Entering the user interface")
        print("\n")
        print("Type in the name of the image that needs to be tested. When done, type END to end the program")
        final = 0
        while final == 0:
            name = input("Image Name")
            if name == "END":
                final = 1
                print("Thank You!!!")
            else:
                img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
                tr = np.reshape(img,(1,240,480,1))
                tr = tr/255
                er1=sess.run([y_h], feed_dict={X: tr})
                fi = np.argmax(er1[0][0])
                if fi == 0:
                    print("Cracked")
                if fi == 1:
                    print("Uncracked")
        

        ########
        return train_accuracy, W1, W2,er2
ta, W1, W2, pp = model(train_vectors, train_labels,test_vectors,test_labels)
print(ta)
##testing
#(m, r, s, t) = test_vectors.shape
#q = test_labels.shape[1]
#X, Y = create_placeholders(r, s, t, q)
#final = forward_propagation(X,W1,W2)
#init_op = tf.initialize_all_variables()
#with tf.Session() as sess:
#	sess.run(init_op)
#	feed_dict = {X:test_vectors}
#	er=sess.run([final], feed_dict=feed_dict)
