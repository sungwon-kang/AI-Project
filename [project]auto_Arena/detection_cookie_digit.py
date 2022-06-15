import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

import cv2 as cv
    
# 데이터 전처리
def preprocessing(img):   
    img=img.astype(np.float32)
    img=img.reshape(1,28,28,1)
    img=tf.keras.utils.normalize(img,axis=1)
    return img

def show(img):
    plt.imshow(img,cmap='gray'),plt.xticks([]),plt.yticks([])
    plt.show()

# 모폴로지 침식을 한 후가 더 결과가 좋다는 것을 확인
#%%
se = np.uint8([[0,1,0],
               [1,1,1],
               [0,1,0]])

# 색 반전해서 불러오기
img = 255-cv.imread('./TestSample/8.jpg')

b_erosion = cv.erode(img,se,iterations=1)

show(img)
show(b_erosion)

# digit=preprocessing(img)
# erosion=preprocessing(b_erosion)

#%%
# 모델 로드
model=load_model('./cnn_v2.h5')
# x1=model.predict([digit])
# x2=model.predict([erosion])

print(x1)
print(x1.argmax())

print(x2)
print(x2.argmax())
#%%

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
# 혼동행렬을 이용한 성능측정.

#digit 데이터 셋 로드
digit=datasets.load_digits()

#훈련 집합과 테스트 집합을 6:4 비율로 나누어서
#훈련 집합과 레이블은 x, 테스트 집합과 레이블은 y에 저장한다.
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)

# SVM 모델을 이미 선택했기 때문에 검증 집합이 없다.
# 학습 후 예측.
s=svm.SVC(gamma=0.001)
s.fit(x_train,y_train)
res=s.predict(x_test)

# 혼동 행렬 구함 ( 예측 i , 실제 j )
conf=np.zeros((10,10))          #10x10 0으로 채운 행렬 생
#print(conf)           		 # 0으로 채운 10x10 행렬이 출력된다.

for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
    conf[res[i]][y_test[i]]+=1 	 #res[i]측정한 값, y_test[i]실제 값 위치에 +1
    
print(conf)		# 출력, 대각선 부분이 예측과 실제값이 일치한 부분이다.

# 정확률 측정하고 출력
no_correct =0
for i in range(10):
    no_correct+=conf[i][i] # 혼동행렬의 대각선 부분을 모두 더한다.

accuracy = no_correct/len(res) # 모두 더한 값에 예측값 수을 나누면 정확도를 구할 수 있다.
print(accuracy*100,"%")