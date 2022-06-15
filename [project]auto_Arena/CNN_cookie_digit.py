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

imgs=[]
x_train=np.array((10,28,28),dtype='float32')

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


def load_imgs(path):
    
    for i in os.listdir('./'+path+'/'):
        path = './'+path+'/'+i 

        img=cv.imread(path, cv.IMREAD_GRAYSCALE)
        img=preprocessing(img)
        
        print(img.shape)
        imgs.append(img)
    
    return imgs
        


#%%
y_train=np.array([0,1,2,3,4,5,6,7,8,9])
y_val=np.array([0,1,2,3,4,5,6,7,8,9])

x_train=np.array(load_imgs('testSample'))
x_val=np.array(load_imgs('valSample'))
# 부류를 원핫코드로 변환
y_train=tf.keras.utils.to_categorical(y_train,10)
y_val=tf.keras.utils.to_categorical(y_val,10)
#%%

# 신경망 모델 설계
# [C-C-P] 구조를 가진 빌딩블럭 2개로 구성된 모델
# 과잉적합을 해소하는 드롭아웃을 이용해 세대마다 가중치 25%를 임의의 가중치를 불능으로 만든다.
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

# 특징 맵을 일렬로 펼치는 함수
cnn.add(Flatten())

# 퍼셉트론 은닉층
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10,activation='softmax'))

# 신경망 모델 학습
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

generator=ImageDataGenerator(width_shift_range=0.3,height_shift_range=0.3,rescale=0.7)
hist = cnn.fit_generator(generator.flow(x_train, y_train, batch_size=128),
                         epochs=50, validation_data=(x_val, y_val),verbose=1)

# 학습된 모델 저장
cnn.save('cnn_v3.h5')
#%%
# 신경망 모델 정확률 평가
# res=cnn.evaluate(x_test,y_test,verbose=0)
# print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()


