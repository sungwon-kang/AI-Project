import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1은 CPU, 나머지 번호는 GPU
#%%
# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)= ds.mnist.load_data()

# 훈련 집합과 테스트 집합의 구조를 변환하고 정규화, 28x28x1 2차원 구조
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

# 부류를 원핫코드로 변환
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)


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
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 학습된 모델 저장
cnn.save('cnn_v2.h5')

# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)
#%%
import cv2 as cv
from PIL import Image

# 이미지 32x32으로 변경
# img=Image.open('TestSample/7.jpg')
# img=img.resize((28,28))
# img=np.array(img)

img =cv.imread('7.jpg',cv.IMREAD_GRAYSCALE)
img=cv.resize(img, (28,28))
img=img.astype('float32')
img=img.reshape(1,28,28,1)
               
img=tf.keras.utils.normalize(img,axis=1)

#%%
model=load_model('../cnn_v2.h5')
x=model.predict([img])
 
print(x)
# cv.imshow('test',gray)

# cv.waitKey()
# cv.destroyAllWindows()


