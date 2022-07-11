import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from skimage import morphology
from matplotlib import pyplot as plt

import time
#%%
class Imageprocessor:
    
    ImgSize=28
    show_flag=True
    Data_type='float32'
    def __init__(self):
        # 모폴로지 구조
        self.se = np.uint8([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
        # self.se = np.uint8([
        #             [0, 0, 1, 0, 0],
        #            [0, 1, 1, 1, 0],
        #            [1, 1, 1, 1, 1],
        #            [0, 1, 1, 1, 0],
        #            [0, 0, 1, 0, 0]
        #            ])
    def PILtoCV(self, img):
        isPIL=str(type(img)) == str("<class 'PIL.Image.Image'>")
        if isPIL == True: 
            np_img=np.array(img)
            cv_img=cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
            img=cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        return img
     
    def CVtoPIL(self, img):
        isCV = str(type(img)) == str("<class 'numpy.ndarray'>")
        if isCV == True:
            cv_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
        return img
    
    def show(self, img):
        if self.show_flag == True:
            plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
            plt.show()
        
    def crop(self, lst_cvimgs, cut, crop_n, crop_fx1, crop_fx2, y2=28):
        
        c=cut                   # 원본 이미지 절단 
        n=crop_n                # 생성할 분할 이미지 수 (1/n)
        fx1=crop_fx1            # 분할 조절
        fx2=crop_fx2
        
        
        listofImgs=[]
        for i in range(len(lst_cvimgs)):
            # OpenCV -> PIL
            img = self.CVtoPIL(lst_cvimgs[i])
            
            #이미지의 크기 출력
            width, height = img.size
            print('이미지 크기 (w, h):', width, height)
            width=width/c

            # 이미지 자르기 crop함수 이용 ex. crop(left, up, rigth, down)
            wn =  width/n
            croppedImgs=[]
            for i in range (n):
                x1 = wn*i+fx1
                x2 = wn*(i+1)+fx2
                
                croppedImage=img.crop(( x1, 0, x2, y2))
                print("잘려진 사진 크기 :",croppedImage.size)    
                croppedImgs.append(croppedImage)
            
            listofImgs.append(croppedImgs)
            
        return listofImgs
    
    def resize(self, gray):
        n=self.ImgSize
        # cv_img가 PIL이 아닌 경우 PIL -> OpenCV
        gray=self.PILtoCV(gray)

        height, width = gray.shape[0], gray.shape[1]
        print('height :',height, 'width :',width)
        
        f = n-height+n-width
        # 이미지 확대, 축소에 따라 보간법을 다르게 적용
        
        if f<0:
            # 축소
            gray = cv.resize(gray, (n, n), interpolation = cv.INTER_AREA) 
        elif f>0:
            # 확대
            gray = cv.resize(gray, (n, n), interpolation = cv.INTER_LINEAR)
        
        self.show(gray)
        return gray
    
    # 영상 이진화
    def binary_img(self, gray):
        t, bin_img = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        self.show(bin_img)
        return bin_img
    
    # 픽셀 이하 연결요소를 삭제
    def remove_PixelFewer(self, bin_img):
        
        # bool 형 배열로 변환
        bool_img=np.array(bin_img, bool)
        
        # 최소 100개 이상 픽셀인 연결 요소는 남기고 나머지 픽셀들은 모두 삭제
        cleaned_img = morphology.remove_small_objects(bool_img, min_size=100, connectivity=1)
        
        # 이진 영상으로 변환
        gray_img=np.array(cleaned_img, dtype='uint8')
        self.show(gray_img)
        return gray_img
    
    # 침식
    def morphology(self, gray):
        mop_img = cv.erode(gray, self.se, iterations=1)
        self.show(mop_img)
        return mop_img
    
    # 흑백 반전
    def inversion_img(self, img):
        inv = img.copy()
        inv_img = 255-inv
        self.show(inv_img)
        return inv_img

    def preprocessing(self, img):
        gray= self.resize(img)
        bin_img = self.binary_img(gray)
        cleaned_img=self.remove_PixelFewer(bin_img)        
        mop_img = self.morphology(cleaned_img)
        
        n = self.ImgSize
        img = mop_img.reshape(n, n, 1)
        img = tf.keras.utils.normalize(img, axis=1)
        
        return img
    
    def listToArray(self, lst, Type):
        return np.array(lst, dtype=Type)
        
    def load_img(self, path, filename, inversion):
        fpath = './'+path+'/'+filename

        img = cv.imread(fpath, cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 흑백 반전 [배경이 흰색이고 숫자가 검은색인 경우 필요]
        if inversion == True:
            gray = self.inversion_img(gray)

        proceed_img = self.preprocessing(gray)
        return np.array(proceed_img, dtype=self.Data_type)
    
    def load_imgs(self, path, inversion=False):
        imgs = []
        unloadable=['prcd','cap_']
        
        for img in os.listdir(path+'/'):
            if img[:4] not in unloadable:
                print(img)
                proceed_img=self.load_img(path, img, inversion)
                imgs.append(proceed_img)
             
        print(path," - 작업 끝")
        # 응답없음 방지
        time.sleep(0.3)
        return np.array(imgs,dtype=self.Data_type)