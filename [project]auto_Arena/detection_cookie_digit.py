import numpy as np
import cv2 as cv
import Imageprocessor as IP
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

ip=IP.Imageprocessor()
#%%
img=cv.imread('./3_CaptureSample/test.png')
ip.crop(cv_img=img, init_n=2, img_n=5, crop_fx1=-4, crop_fx2=0)
#%%
Imgs=[]
for i in range (5):
    img=cv.imread('./3_CaptureSample/croppedSample/test'+str(i)+'.png',cv.COLOR_RGB2BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ip.show(gray)
    Imgs.append(gray)


proceed_imgs=[]

for img in Imgs:
    img=ip.preprocessing(img)
    proceed_imgs.append(img)

imgs=np.array([proceed_imgs],dtype='float32')
#%%
# 혼동 행렬 구함 ( 예측 i , 실제 j )
conf=np.zeros((10,10))          #10x10 0으로 채운 행렬 생성
cnn = load_model('0_models/cnn_v4.h5')
res=cnn.predict(imgs)

y=np.array([0,0,3,4,4,0])

for img in imgs:
    ip.show(img)
    
for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
    conf[np.argmax(res[i])][y[i]]+=1 	 #res[i]측정한 값, y_test[i]실제 값 위치에 +1
    
print(conf)		# 출력, 대각선 부분이 예측과 실제값이 일치한 부분이다.

# 정확률 측정하고 출력
no_correct =0
for i in range(10):
    no_correct+=conf[i][i] # 혼동행렬의 대각선 부분을 모두 더한다.

accuracy = no_correct/len(res) # 모두 더한 값에 예측값 수을 나누면 정확도를 구할 수 있다.
print(accuracy*100,"%")