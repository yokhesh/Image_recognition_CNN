import cv2;
import numpy as np;
import tensorflow as tf
from tensorflow import keras

#from matplotlib import pyplot as plt
a = 1
#Extracting images of cracked glass
for i in range(14):
    i = i+1
    b = str(a)
    img = None
    img = cv2.imread(b+'.jpg')
    #cv2.imshow('image',img)
    #img1 = cv2.imread('2.jpg');
    v = None
    v = img.reshape((img.shape[0]*img.shape[1]*img.shape[2],1))
    if i == 2:
        con = np.concatenate((v_prev, v), axis=1)
        
    if i > 2:
        con = np.concatenate((con, v), axis=1)
    v_prev =None
    v_prev = v
    a = a+1
a1 = 1
# Doubling the image vectors
con = np.concatenate((con, con), axis=1)
# Creating test vectors for cracked glass
#y = np.repeat(1,14)
y = np.full((28, 2), [1,0])
#Extracting images of uncracked glass
for i in range(14):
    i = i+1
    b = str(a1)
    
    img = None
    img = cv2.imread(b+'_p.jpg')
    v = None
    v = img.reshape((img.shape[0]*img.shape[1]*img.shape[2],1))
    if i == 2:
        con1 = np.concatenate((v_prev, v), axis=1)
        
        
    if i > 2:
        con1 = np.concatenate((con1, v), axis=1)
        
    v_prev =None
    v_prev = v
    a1 = a1+1
# Doubling the image vectors
con1 = np.concatenate((con1, con1), axis=1)
# Combining the cracked and uncracked features
con = np.concatenate((con, con1), axis=1)
#Normalize vectors
train_vectors = con/255
# Creating test vectors for uncracked glass
#yp = np.repeat(0,14)
yp = np.full((28, 2), [0,1])
train_labels = np.concatenate((y, yp), axis=0)
y = train_labels
#Tensor flow

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-5), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
yu = model.fit(train_vectors.T, train_labels, epochs=20)

################################################Testing###################
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
        img = cv2.imread(name)
        tr = img.reshape((img.shape[0]*img.shape[1]*img.shape[2],1))
        tr = tr/255
        test_pred = model.predict(tr.T)
        fi = np.argmax(test_pred)
        if fi == 0:
            print("Cracked")
        if fi == 1:
            print("Uncracked")
        
