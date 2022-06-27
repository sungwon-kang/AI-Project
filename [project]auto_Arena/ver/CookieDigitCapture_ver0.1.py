import cv2 as cv
import numpy as np
import tensorflow as tf
import Imageprocessor as ip

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import sys

import pyautogui    # mouse, keyboard 관련 도구함 끌어오기
import pyperclip    # 클립보드에 저장 혹은 클립보드에서 불러오기 위함

from tensorflow.keras.models import load_model
IP=ip.Imageprocessor()

# 임시 라이브러리
from datetime import datetime 

import os
#%%
# 처리 순서
# 1. 이미지 재조정
# 2. 전처리
# 3. 예측
# 4. 숫자연결
# 5. 반환
'''
	Ver0.1 시나리오
	[캡처 기능]
		1. 임의의 마우스 위치에서 임의의 단축키를 눌러 좌표를 찍는다.
		2. (2)의 행동을 한번 더 찍어 (x1,x2,y1,y2)를 얻고 임의의 단축키를 눌러 숫자 이미지를 캡처한다.
        3. 종료 단축키를 누르지 않을 시, 1번을 반복 수행할 수 있다. 종료 단축키를 누르면 캡처 기능이 멈춘다.
        
	[분할 이후 전처리 기능]
		1. 얻은 이미지를 한 자릿수로 인식할 수 있도록 분할한다.
		2. 분할한 이미지들을 저장한다.
		3. 분할한 이미지들을 모델의 입력 데이터 형태로 변화하고 인식을 수행한다.

	[기록]
		1. 한 자릿수 분할한 이미지와 인식의 결과를 기록한다. ( argmax 후 .txt )
		2. 저장된 이미지와 인식 결과를 가지고 정확도 계산에 쓴다.
    
    [나가기]
'''
#%%
class DigitCapture(QMainWindow):
    
    x1=-1
    x2=-1
    
    y1=-1
    y2=-1
    
    keyFlag=True
    tmp_DateName=None
    tmp_fileName=None
    tmp_predicted_lst=[]
    
    saved_p1_cord=[]
    saved_p2_cord=[]
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        self.setGeometry(200, 200, 800, 250)
        self.setFixedSize(800, 250)
        self.setWidget()
        
        self.cnn=load_model('./0_models/cnn_v4.h5')
        self.ext='.jpg'
        # 임시 변수, 삭제할 예정
        self.tmp_imgpath='./3_CaptureSample/'
        self.tmp_savepath='./3_CaptureSample/'       
        
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_A, Qt.Key_E, Qt.Key_C]
        key=e.key()
        
        if(self.keyFlag==True and key in printable):
            if key == Qt.Key_Q:
                self.x1, self.y1 = pyautogui.position()
                print("위치 값 : ({}, {})\n".format(self.x1,self.y1))
            
            elif key == Qt.Key_W:
                self.x2, self.y2 = pyautogui.position()
                print("위치 값 : ({}, {})\n".format(self.x2,self.y2))
               
            elif key == Qt.Key_A: 
                self.saved_p1_cord.append((self.x1,self.y1))
                self.saved_p2_cord.append((self.x2,self.y2))
                
                print(self.saved_p1_cord)
                print(self.saved_p2_cord)
            
            elif key == Qt.Key_C:
                self.saved_p1_cord.clear()
                self.saved_p2_cord.clear()
                print("list all clear")
            elif key == Qt.Key_E:
                l1=len(self.saved_p1_cord)
                l2=len(self.saved_p2_cord)
                if( ( l1>0 and l2 > 0) and (l1 == l2) ):
                    for i in range(l1):
                        (x1,y1)=self.saved_p1_cord[i]
                        (x2,y2)=self.saved_p2_cord[i]
                        print("두 위치 : ({}, {}), ({}, {})".format(x1,y1,x2,y2))

                        sorted_x_value = sorted([x1, x2])
                        sorted_y_value = sorted([y1, y2])
                        width = sorted_x_value[1] - sorted_x_value[0]
                        height = sorted_y_value[1] - sorted_y_value[0]

                        x_re = sorted_x_value[0] - 20
                        y_re = sorted_y_value[0] - 20
                        width_re = width + 40
                        height_re = height + 40
                                                  
                        print("영역 : ({}, {}, {}, {})\n".format(x_re, y_re, width_re, height_re))
                        pyperclip.copy("({}, {}, {}, {})".format(x_re, y_re, width_re, height_re))
                
                        if 'x_re' in locals(): # 메모리 변수 속에 있는 것을 불러오기. 따라서 x_re이 있어야 함.
                            self.tmp_DateName=datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
                            self.CaptureScreen(sorted_x_value, sorted_y_value, width, height)
                            # print(self.tmp_DateName)
                    
                    
                
    def CaptureScreen(self, sorted_x_value, sorted_y_value, width, height):
        # ver0.1 아닌 버전은 삭제할 것
        # E를 누를 시 저장된 좌표에 맞춰 캡쳐
        
        self.tmp_fileName = self.tmp_imgpath+self.tmp_DateName+'/'+self.tmp_DateName+self.ext
        
        print(self.tmp_fileName)
        os.mkdir(self.tmp_imgpath+self.tmp_DateName)
        
        pyautogui.screenshot(self.tmp_fileName ,region=(sorted_x_value[0], sorted_y_value[0], width, height))
        print("저장완료")
        
        # 삭제할 것
        self.processingFunction()
        self.predictFunction()
        self.saveFunction()
        
        
    def processingFunction(self):
        # 다이알로그를 이용하거나 상대경로로 지정할 예정
        img=cv.imread(self.tmp_fileName)
        # 입력 박스에서 매개변수를 받아오도록 구현할 예정
        Savepath=str(self.tmp_savepath+self.tmp_DateName)
        cropped_imgs=IP.crop(Savepath, cv_img=img, init_n=1, img_n=7, crop_fx1=-4, crop_fx2=0,y2=28)
        
        
        # 삭제할 것
        i=0
        tmp_Imglist=[]            
        for img in cropped_imgs:
            proceed_img=IP.preprocessing(img,self.tmp_DateName,i)
            
            # 삭제할 것
            tmp_Imglist.append(proceed_img)
            i+=1
        self.proceed_imgs=np.array(tmp_Imglist, dtype=IP.Data_type)
        print(self.proceed_imgs.shape)
        print("전처리 완료")
        
    def predictFunction(self):
        
        if len(self.proceed_imgs)==0:
            print("등록된 이미지가 없습니다.")
           
        else:
            res=self.cnn.predict(self.proceed_imgs)           
            num=0
            n_digit = len(res)-1
            for digit in res:
                predicted_num=np.argmax(digit)
                
                num+=predicted_num*(10**n_digit)
                n_digit=n_digit-1
                
                self.tmp_predicted_lst.append(predicted_num)
                print(predicted_num)
            
            print(num)
    
    def saveFunction(self):
        with open(str(self.tmp_savepath+'labels.txt'), 'a') as File:
            for i in range (7):
                filename=self.tmp_DateName+'('+str(i)+').jpg'+'\t\t'+str(self.tmp_predicted_lst[i])+'\t\n'
                File.write(filename)
                print('추가됨 :' + self.tmp_DateName+'('+str(i)+').jpg', self.tmp_predicted_lst[i])
            
            File.write("===========================\n")
            self.tmp_predicted_lst.clear()
            File.close()
            
    # btn_guide(사용법) 이벤트 함수    
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', '')
    
    #btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        self.close()
    
    def setWidget(self):
        # 프레임과 레이아웃 선언
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0,800,100)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
    
        frame_bottom = QFrame(self)
        frame_bottom.setGeometry(0,100,800,150)
        frame_bottom.setFrameShape(QFrame.Box | QFrame.Plain)
    
        main_layout=QVBoxLayout()
        layout_top = QVBoxLayout()
        layout_bottom = QVBoxLayout()
        
        # 버튼 및 라벨 설정
        btn_processing = QPushButton('분할 및 전처리', self)
        btn_predict = QPushButton('예측', self)
        # btn_recognition = QPushButton('3. 손 인식하기', self)
        # btn_compose = QPushButton('4. 이미지 합성', self)
        
        btn_guide = QPushButton('사용법', self)
        # btn_save = QPushButton('이미지 저장', self)        
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        btn_processing.setGeometry(10, 10, 100, 30)
        btn_predict.setGeometry(110, 10, 100 , 30)
        # btn_recognition.setGeometry(210, 10, 100, 30)
        # btn_compose.setGeometry(310, 10, 100, 30)
        
        btn_guide.setGeometry(490, 10, 100, 30)
        # btn_save.setGeometry(590, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_processing)
        layout_top.addWidget(btn_predict)
        # layout_top.addWidget(btn_recognition)
        # layout_top.addWidget(btn_compose)
        layout_top.addWidget(btn_guide)
        # layout_top.addWidget(btn_save)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
        # 각 버튼에 콜백함수 연결
        btn_processing.clicked.connect(self.processingFunction)
        btn_predict.clicked.connect(self.predictFunction)
        # btn_recognition.clicked.connect(self.recognitionFunction)
        # btn_compose.clicked.connect(self.composeFunction)
        # btn_save.clicked.connect(self.saveImgFunction)
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
    
app = QApplication(sys.argv)
win = DigitCapture()
win.show()
app.exec_()



