import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import sys
#%%
img=cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

# 해당 영상이 없을 시 오류 메세지를 출력하고 종료한다.
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 오츄 알고리즘을 이용하여 최적의 문턱값을 구한다.
# 문턱값과 문턱값으로 이진화된 영상을 반환한다.
t, bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#xticks()와 yticks()은 그래프 x,y축 간격마다 있는 눈금에 대한 메소드이다.
#[]을 인자로 전달할 시 눈금을 삭제한다.
#이진화된 영상을 gray scale로 출력한다.
plt.imshow(bin_img,cmap='gray'), plt.xticks([]),plt.yticks([])
plt.show()

#최적 임계값 출력해보기.
#print(t) # 출력값 124.0

# 영상을 [행의 크기]//2와 [행의 크기]//2+1으로 잘라내어 정사각형 모양으로 출력.
# +1을 한 이유는 0부터 시작하는 첫 인덱스에 맞춰 정사각형으로 잘라낸다.
b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]

# 정사각형 영상 크기 출력해보기
print(bin_img.shape[0]//2,bin_img.shape[0],0,bin_img.shape[0]//2+1)
plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

# 모폴로지 연산에 필요한 구조요소 정의
# 데이터형은 8비트(0~255) 부호없는 정수형
se = np.uint8([[0,0,1,0,0],
               [0,1,1,1,0],
               [1,1,1,1,1],
               [0,1,1,1,0],
               [0,0,1,0,0]])

# 팽창: cv.dilate(영상, 구조요소, 반복 횟수)
# 출력: 전체적으로 영상의 밝은부분이 두꺼워졌다. 
b_dilation = cv.dilate(b,se,iterations=1) 
plt.imshow(b_dilation,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

# 침식: cv.erode(영상, 구조요소, 반복 횟수)
# 출력: 이어져있던 밝은부분이 끊어지고 두꺼운 부분들이 얇아졌다.
b_erosion = cv.erode(b,se,iterations=1) 
plt.imshow(b_erosion,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

# 닫기: 팽창 후 침식을 수행하는 모폴로지 연산
# 팽창함수(dilate)으로 반환된 영상을 바로 침식함수(erode)에 인자로 전달하여 닫기를 수행.
# 출력: 밝은 부분 중 얇은 부분만 두꺼워졌다.
b_closing=cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1)
plt.imshow(b_closing,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

# 열림: 침식 후 팽창을 수행하는 모폴로지 연산
# 침식함수(erode)으로 반환된 영상을 바로 팽창함수(dilate)에 인자로 전달하여 열기를 수행.
# 출력: 밝은 부분 중 얇은 부분만 끊어졌다.
b_opening=cv.dilate(cv.erode(b,se,iterations=1),se,iterations=1)
plt.imshow(b_opening,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

