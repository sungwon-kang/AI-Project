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
    
    Data_type='float32'
    ImgSize=28
    show_flag=True
    def __init__(self):
        # 모폴로지 구조
        self.se = np.uint8([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
     
    def PILtoCV(self, PIL_img):
         np_img=np.array(PIL_img)
         cv_img=cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
         cv_img=cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
         return cv_img
     
    def CVtoPIL(self, CV_img):
        cv_img = cv.cvtColor(CV_img, cv.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(cv_img)
        return PIL_img
    
    def show(self, img):
        if(self.show_flag==True):
            plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
            plt.show()
        
    def crop(self, Savepath, cv_img, init_n, img_n, crop_fx1, crop_fx2, y2):
        
        i=init_n             # 초기 이미지 너비 조정
        n=img_n             # 이미지 분할 수
        fx1=crop_fx1           # 분할 조절
        fx2=crop_fx2
       
        
        # OpenCV -> PIL
        img = self.CVtoPIL(cv_img)
        
        #이미지의 크기 출력
        width, height = img.size
        print('이미지 크기 (w, h):', width, height)
        width=width/i
        # print(width)
        
        # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
        croppedImgs=[]
        wn =  width/n
        
        # 삭제할 것
        splited_path=Savepath.split(sep='/')
        subpath=splited_path[1]
        subfolder=splited_path[2]
        mainfolder=splited_path[3]
        print(splited_path)
        for i in range (n):
            x1 = wn*i+fx1
            x2 = wn*(i+1)+fx2
            
            croppedImage=img.crop(( x1 , 0, x2, y2))
            print("잘려진 사진 크기 :",croppedImage.size)    
            croppedImgs.append(croppedImage)
            
            # 삭제할 것
            rpath='./'+subpath+'/'+subfolder+'/'+mainfolder+'/'+mainfolder+'['+str(i)+']'+'.jpg'
            croppedImage.save(rpath)
        
        return croppedImgs
    
    def resize(self, gray):
        n=self.ImgSize
        # cv_img가 PIL이 아닌 경우 PIL -> OpenCV
        if str(type(gray)) == str("<class 'PIL.Image.Image'>"):
            gray=self.PILtoCV(gray)

        height, width = gray.shape[0], gray.shape[1]
        print('height :',height, 'width :',width)
        
        f = n-height+n-width
        if f<0:
            gray = cv.resize(gray, (n, n), interpolation = cv.INTER_AREA)
            
        elif f>0:
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
        
        # 최소 100개 이상 픽셀인 연결 요소를 남기고 나머지 모두 삭제
        cleaned_img = morphology.remove_small_objects(bool_img, min_size=100, connectivity=1)
        
        # 다시 이진 영상으로 변환
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

    # 위 과정들을 순서대로 수행
    # 매개변수 삭제할 것
    
    def tmp_SaveImg(self, mop_img, tmp_DateName, i):
        # 삭제할 것
        tmp_path='./3_CaptureSample/2022-06-29/'+tmp_DateName+'/prcd_'+tmp_DateName+'['+str(i)+'].jpg'
        plt.imsave(tmp_path,mop_img,cmap="gray")
        
    def preprocessing(self, img): 
    # , tmp_DateName, i):
    
        gray= self.resize(img)
        bin_img = self.binary_img(gray)
        cleaned_img=self.remove_PixelFewer(bin_img)        
        mop_img = self.morphology(cleaned_img)
        
        # self.tmp_SaveImg(mop_img, tmp_DateName, i)
        
        n = self.ImgSize
        img = mop_img.reshape(n, n, 1)
        img = tf.keras.utils.normalize(img, axis=1)
        
        return img
    

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
        time.sleep(0.5)
        return np.array(imgs,dtype=self.Data_type)
        