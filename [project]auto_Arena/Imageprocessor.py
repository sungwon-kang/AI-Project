import os
import cv2 as cv
import numpy as np
from PIL import Image

import tensorflow as tf
from matplotlib import pyplot as plt
#%%
class Imageprocessor:
    
    def __init__(self):
        # 모폴로지 구조
        self.se = np.uint8([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
        
    def crop(self, cv_img, init_n, img_n, crop_fx1,crop_fx2):
        
        i=init_n             # 초기 이미지 너비 조정
        n=img_n             # 이미지 분할 수
        fx1=crop_fx1           # 분할 조절
        fx2=crop_fx2
        
        # OpenCV -> PIL
        cvt_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(cvt_img)
        
        #이미지의 크기 출력
        width, height = img.size
        
        print('이미지 크기 (w, h):', width, height)
        width=width/i
        # print(width)
        
        # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
        wn =  width/n
        croppedImgs=[]
        for i in range (n):
            x1 = wn*i+fx1
            x2 = wn*(i+1)+fx2
            
            croppedImage=img.crop(( x1 , 0, x2, height))
            print("잘려진 사진 크기 :",croppedImage.size)
            croppedImgs.append(croppedImage)
            croppedImage.save('./3_CaptureSample/croppedSample/test'+str(i)+'.jpg')
        
        return croppedImgs
    
    def resize(self, img):
        n=28
        
        # PIL -> OpenCV
        img=np.array(img)
        img=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        height, width = img.shape
        
        f = n-height+n-width
        if f<0:
            img = cv.resize(img, (n, n), interpolation = cv.INTER_AREA)
            
        elif f>0:
            img = cv.resize(img, (n, n), interpolation = cv.INTER_LINEAR)
        
        return img

    # 침식
    def morphology(self, gray):
        mop_img = cv.erode(gray, self.se, iterations=1)
        return mop_img

    # 영상 이진화
    def binary_img(self, gray):
        t, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return bin_img

    # 흑백 반전
    def inversion_img(self, img):
        inv = img.copy()
        inv_img = 255-inv
        return inv_img

    # 위 과정들을 순서대로 수행
    def preprocessing(self, img):
    
        resized_img= self.resize(img)
        bin_img = self.binary_img(resized_img)
        mop_img = self.morphology(bin_img)
        
        self.show(mop_img)
        
        img = mop_img.reshape(28, 28, 1)
        img = tf.keras.utils.normalize(img, axis=1)
        
        return img


    def show(self, img):
        plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
        plt.show()

    
    def load_imgs(self, path, inversion):
        imgs = []

        for i in os.listdir('./'+path+'/'):
            fpath = './'+path+'/'+i

            img = cv.imread(fpath, cv.COLOR_RGB2BGR)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # 흑백 반전 [배경이 흰색이고 숫자가 검은색인 경우 필요]
            if inversion == True:
                gray = self.inversion_img(gray)
            
            # self.show(gray)
            proceed_img = self.preprocessing(gray)
            
            imgs.append(proceed_img)
        return imgs
