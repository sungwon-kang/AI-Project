import numpy as np
import cv2 as cv
import Imageprocessor as ip
import os
from tensorflow.keras.models import load_model

IP=ip.Imageprocessor()
cnn = load_model('0_models/cnn_v5.h5')

#%%
n=7
img=cv.imread('./3_CaptureSample/test.jpg')
IP.crop(cv_img=img, init_n=1, img_n=n, crop_fx1=-3, crop_fx2=0,y2=28)
#%%

# 여러 이미지
Imgs=[]
Imgs=IP.load_imgs('3_CaptureSample/croppedSample', False)
Imgs=np.array(Imgs,dtype='float32')
print(Imgs.shape)
#%%
# 예측
res=cnn.predict(img)

# 부류
y=np.array([1,1,1,0,3,9,1])
#%%


conf=np.zeros((10,10))          #10x10 0으로 채운 행렬 생성
for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
    conf[np.argmax(res[i])][y[i]]+=1 	 #res[i]측정한 값, y_test[i]실제 값 위치에 +1
    
print(conf)		# 출력, 대각선 부분이 예측과 실제값이 일치한 부분이다.

# 정확률 측정하고 출력
no_correct =0
for i in range(10):
    no_correct+=conf[i][i] # 혼동행렬의 대각선 부분을 모두 더한다.

accuracy = no_correct/len(res) # 모두 더한 값에 예측값 수을 나누면 정확도를 구할 수 있다.
print(accuracy*100,"%")