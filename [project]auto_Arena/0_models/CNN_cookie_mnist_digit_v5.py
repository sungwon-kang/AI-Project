import numpy as np
import tensorflow as tf
import Imageprocessor as ip
import tensorflow.keras.datasets as ds

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1은 CPU, 나머지 번호는 GPU
tf.__version__
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
#%%
IP=ip.Imageprocessor()
(x_train_mnist,y_train_mnist),(x_test_mnist,y_test_mnist)= ds.mnist.load_data()
# # MNIST 훈련 집합과 테스트 집합의 구조를 변환하고 정규화, 28x28x1 2차원 구조
x_train_mnist=x_train_mnist.reshape(60000,28,28,1)
x_test_mnist=x_test_mnist.reshape(10000,28,28,1)
x_train_mnist=x_train_mnist.astype(np.float32)/255.0
x_test_mnist=x_test_mnist.astype(np.float32)/255.0

# cookie 훈련 집합과 검증 집합
x_train_cookie=IP.load_imgs('1_trainSample',True)
x_test_cookie=IP.load_imgs('2_valSample', False)

y_train_cookie=np.array([0,1,2,3,4,5,6,7,8,9,0])
y_test_cookie=np.array([0,1,2,3,4,5,6,7,8,9,0,0])
#%%
folder='./3_captureSample/'
year='2022'
mon_day=['06-27','06-28','06-29','07-03','07-04','07-05']
#%%
n=1
for date in mon_day:
    path = '{}{}-{}'.format(folder, year, date)
    print(path)
    
    for Files in os.listdir(path+'/'):
        FilePath = path + '/' + Files
        print(FilePath)
        if Files == "labels.txt":
            f = open(FilePath,'r')
            
            lines = f.readlines()
            y=[]
            for line in lines:
                line = line.strip()
                if line[-1:]!='=':
                    y.append(int(line[-1:]))
                    
            # print(y)
            y_train_cookie=np.concatenate([y_train_cookie, np.array(y)])
            print("레이블 불러오기 끝")
                    
        else:
            x_train_cookie=np.concatenate([x_train_cookie, IP.load_imgs(FilePath)])
            print(str(n)+"번째 폴더 작업 끝")
            n+=1
    n=0

#%%
print(x_test_cookie.shape)  
print(y_test_cookie.shape)
print(x_train_cookie.shape)
print(y_train_cookie.shape)    
#%%
#훈련 집합과 테스트 집합을 6:4 비율로 나누어서
#훈련 집합과 레이블은 x, 테스트 집합과 레이블은 y에 저장한다.
tmp_x_train_cookie, tmp_x_test_cookie, tmp_y_train_cookie, tmp_y_test_cookie=train_test_split(x_train_cookie,
                                                                                           y_train_cookie,
                                                                                           train_size=0.6)
x_train_cookie=tmp_x_train_cookie.copy()
y_train_cookie=tmp_y_train_cookie.copy()

x_test_cookie=tmp_x_test_cookie.copy()
y_test_cookie=tmp_y_test_cookie.copy()

#%%
# 부류를 원핫코드로 변환
y_train_mnist=tf.keras.utils.to_categorical(y_train_mnist,10)
y_test_mnist=tf.keras.utils.to_categorical(y_test_mnist,10)

y_train_cookie=tf.keras.utils.to_categorical(y_train_cookie,10)
y_test_cookie=tf.keras.utils.to_categorical(y_test_cookie,10)

# 배열 합치기
# 이미지
x_train=np.concatenate([x_train_mnist,x_train_cookie])
x_test=np.concatenate([x_test_mnist,x_test_cookie])

# 부류
y_train=np.concatenate([y_train_mnist,y_train_cookie])
y_test=np.concatenate([y_test_mnist,y_test_cookie])
#%%

# 신경망 모델 설계
# [C-C-P] 구조를 가진 빌딩블럭 2개로 구성된 모델
# 과잉적합을 해소하는 드롭아웃을 이용해 세대마다 가중치 25%를 임의의 가중치를 불능으로 만든다.
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(128,(3,3),activation='relu'))
cnn.add(Conv2D(128,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))
cnn.add(Dropout(0.25))

# 특징 맵을 일렬로 펼치는 함수
cnn.add(Flatten())

# 퍼셉트론 은닉층
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10,activation='softmax'))

# 신경망 모델 학습
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)

generator=ImageDataGenerator(width_shift_range=0.2)
hist = cnn.fit(generator.flow(x_train, y_train, batch_size=256), epochs=20, validation_data=(x_test, y_test), 
               callbacks=[es], verbose=1)
    
#%%
# 학습된 모델 저장
cnn.save('cnn_v5_1.h5')
print(cnn.summary())
#%%
# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)
#%%
# 혼동 행렬 구함 ( 예측 i , 실제 j )
cnn=load_model('./0_models/cnn_v5_1.h5')

conf=np.zeros((10,10))          #10x10 0으로 채운 행렬 생
res=cnn.predict(tmp_x_test_cookie)

for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
    conf[np.argmax(res[i])][tmp_y_test_cookie[i]]+=1 	 #res[i]측정한 값, y_test[i]실제 값 위치에 +1
    
print(conf)		# 출력, 대각선 부분이 예측과 실제값이 일치한 부분이다.

# 정확률 측정하고 출력
no_correct =0
for i in range(10):
    no_correct+=conf[i][i] # 혼동행렬의 대각선 부분을 모두 더한다.

accuracy = no_correct/len(res) # 모두 더한 값에 예측값 수을 나누면 정확도를 구할 수 있다.
print(accuracy*100,"%")
#%%
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


