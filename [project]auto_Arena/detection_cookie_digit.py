import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

import cv2 as cv
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1은 CPU, 나머지 번호는 GPU

#%%

x_train=np.array((10,28,28),dtype='float32')
x_val=np.array((10,28,28),dtype='float32')

def resize(img):
    img=cv.resize(img, (28,28))
    return img

def preprocessing(img):
    se = np.uint8([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    img = cv.erode(img,se,iterations=1)
    
    img=img.reshape(28,28,1)
    img=tf.keras.utils.normalize(img,axis=1)
    
    return img

def show(img):
    plt.imshow(img,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()


def load_imgs(path,a):
    imgs=[]
    
    for i in os.listdir('./'+path+'/'):
        fpath = './'+path+'/'+i 

        img=cv.imread(fpath, cv.COLOR_RGB2BGR)
    
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        # t,bin_img=cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        rev=gray.copy()
        gray=a-rev
        
        
        proceed_img=preprocessing(gray)
        
        show(gray)
        
        imgs.append(proceed_img)
    return imgs
        
    
#%%
x_train=np.array(load_imgs('trainSample',255))
x_val=np.array(load_imgs('valSample', 0))

# 부류를 원핫코드로 변환
y_train=np.array([0,1,2,3,4,5,6,7,8,9])
y_val=np.array([0,1,2,3,4,5,6,7,8,9])

y_train=tf.keras.utils.to_categorical(y_train,10)
y_val=tf.keras.utils.to_categorical(y_val,10)



