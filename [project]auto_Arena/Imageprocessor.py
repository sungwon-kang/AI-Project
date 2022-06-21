import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

class Imageprocessor:
    
    def __init__(self):
        # 모폴로지 구조
        self.se = np.uint8([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    
    def resize(self, img):
        img = cv.resize(img, (28, 28))
        return img

    # 침식
    def morphology(self, gray):
        mop_img = cv.erode(gray, self.se, iterations=1)
        return mop_img

    # 영상 이진화
    def Binary_img(self, gray):
        t, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return bin_img

    # 흑백 반전
    def inversion_img(self, img):
        inv = img.copy()
        inv_img = 255-inv
        return inv_img


    def preprocessing(self, img):
        bin_img = self.Binary_img(img)
        mop_img = self.morphology(bin_img)
        
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

            # 흑백 반전
            if inversion == True:
                gray = self.inversion_img(gray)
            
            # self.show(gray)
            proceed_img = self.preprocessing(gray)
            
            imgs.append(proceed_img)
        return imgs
