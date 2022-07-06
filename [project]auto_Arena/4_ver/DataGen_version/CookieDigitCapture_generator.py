import cv2 as cv
import numpy as np
import tensorflow as tf
import Imageprocessor as ip

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import *
import sys

import pyautogui    # mouse, keyboard 관련 도구함 끌어오기
import pyperclip    # 클립보드에 저장 혹은 클립보드에서 불러오기 위함
import time

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
    
    refresh_x=0
    refresh_y=0
    
    gui_x=0
    gui_y=0
    
    Thread_flag=True
    
    keyFlag=True
    tmp_DateName=None
    tmp_fileName=None
    tmp_predicted_lst=[]
    
    saved_p1_cord=[]
    saved_p2_cord=[]
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('Auto Arena Generator')
        self.setGeometry(200, 200, 420, 100)
        self.setFixedSize(420, 100)
        
        self.lb_event = QLabel('사용법을 확인하세요!',self)
        self.setUI()
        
        self.cnn=load_model('./0_models/cnn_v5_1.h5')
        self.ext='.jpg'
        # 임시 변수, 삭제할 예정
        self.tmp_imgpath='./3_CaptureSample/2022-07-05/'
        self.tmp_savepath='./3_CaptureSample/2022-07-05/'       
        
        self.timer=QTimer()
        self.timer.timeout.connect(self.autoCaptureFunction)
        
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_A,
                   Qt.Key_E, Qt.Key_C, Qt.Key_M,
                   Qt.Key_N]
        key=e.key()
        
        if(self.keyFlag==True and key in printable):
            if key == Qt.Key_Q:
                self.x1, self.y1 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x1,self.y1))
                self.setText("위치 입력 (x1, y1) ({}, {})".format(self.x1,self.y1))
            
            elif key == Qt.Key_W:
                self.x2, self.y2 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x2,self.y2))
                self.setText("위치 입력 (x2, y2) ({}, {})".format(self.x2,self.y2))
            
            elif key == Qt.Key_N:
                self.gui_x, self.gui_y = pyautogui.position()
                
                print("프로그램 클릭 위치 값 ({}, {})\n".format(self.gui_x, self.gui_y))
                self.setText("프로그램 위치 ({}, {})\n".format(self.gui_x, self.gui_y))
                
            elif key == Qt.Key_M:
                self.refresh_x, self.refresh_y = pyautogui.position()
                
                print("새로고침 위치 값 ({}, {})\n".format(self.refresh_x, self.refresh_y))
                self.setText("새로고침 위치 ({}, {})\n".format(self.refresh_x, self.refresh_y))
            elif key == Qt.Key_A: 
                self.saved_p1_cord.append((self.x1,self.y1))
                self.saved_p2_cord.append((self.x2,self.y2))
                
                print(self.saved_p1_cord)
                print(self.saved_p2_cord)
                self.setText("리스트 추가됨 "+str(self.saved_p1_cord) + str(self.saved_p2_cord))
            
            elif key == Qt.Key_C:
                self.saved_p1_cord.clear()
                self.saved_p2_cord.clear()
                print("list clear")
                self.setText("리스트 초기화됨")
                
            elif key == Qt.Key_E:
                l1=len(self.saved_p1_cord)
                l2=len(self.saved_p2_cord)
                if( ( l1>0 and l2 > 0) and (l1 == l2) ):
                    for i in range(l1):
                        (x1,y1)=self.saved_p1_cord[i]
                        (x2,y2)=self.saved_p2_cord[i]
                        print("두 위치 ({}, {}), ({}, {})".format(x1,y1,x2,y2))

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
                        self.setText("예측 끝")                            
   
    def CaptureScreen(self, sorted_x_value, sorted_y_value, width, height):
        # ver0.1 아닌 버전은 삭제할 것
        # E를 누를 시 저장된 좌표에 맞춰 캡쳐
        self.tmp_fileName = self.tmp_imgpath+self.tmp_DateName+'/cap_'+self.tmp_DateName+self.ext
        
        print(self.tmp_fileName)
        os.mkdir(self.tmp_imgpath+self.tmp_DateName)
        
        img=pyautogui.screenshot(self.tmp_fileName ,region=(sorted_x_value[0], sorted_y_value[0], width, height))
        self.setText("좌표대로 이미지 저장됨")
        print(img)
        # 삭제할 것
        self.processingFunction()
        self.predictFunction()
        self.saveFunction()
        
        
    def processingFunction(self):
        # 다이알로그를 이용하거나 상대경로로 지정할 예정
        img=cv.imread(self.tmp_fileName)
        # 입력 박스에서 매개변수를 받아오도록 구현할 예정
        Savepath=str(self.tmp_savepath+self.tmp_DateName)
        cropped_imgs=IP.crop(Savepath, cv_img=img, init_n=1, img_n=7, crop_fx1=0, crop_fx2=0,y2=28)
        
        
        # 삭제할 것
        i=0
        tmp_Imglist=[]            
        for img in cropped_imgs:
            proceed_img=IP.preprocessing(img,self.tmp_DateName,i)
            
            # 삭제할 것
            tmp_Imglist.append(proceed_img)
            i+=1
        self.proceed_imgs=np.array(tmp_Imglist, dtype=IP.Data_type)
        self.setText("이미지 전처리 작업 끝")
        
    def predictFunction(self):
        
        if len(self.proceed_imgs)==0:
            self.setText("등록된 이미지가 없습니다.")
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
            
            self.setText("예측된 값 "+str(num))
    
    def saveFunction(self):
        with open(str(self.tmp_savepath+'labels.txt'), 'a') as File:
            for i in range (7):
                filename=self.tmp_DateName+'['+str(i)+'].jpg'+'\t\t'+str(self.tmp_predicted_lst[i])+'\t\n'
                File.write(filename)
                print('추가됨 :' + self.tmp_DateName+'['+str(i)+'].jpg', self.tmp_predicted_lst[i])
            
            File.write("===========================\n")
            self.tmp_predicted_lst.clear()
            File.close()
        
        self.setText("정상적으로 저장완료")
    
    def threadFunction(self):
        ms=1000
        
        if self.Thread_flag==True:
            self.timer.start(610*ms)
            self.Thread_flag=False
        
        else:
            self.timer.stop()
            self.Thread_flag=True
            
    def autoCaptureFunction(self):
        pyautogui.click(self.refresh_x, self.refresh_y)
        time.sleep(1)
        pyautogui.click(self.gui_x,self.gui_y)
        time.sleep(1)
        pyautogui.press("e")
        
    # btn_guide(사용법) 이벤트 함수    
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', '[단축키]\n'+
                                '  <Q>를 누를 시 마우스 포인터 기준으로 x1, y1이 저장됩니다.\n'+
                                '  <W>를 누를 시 마우스 포인터 기준으로 x2, y2이 저장됩니다.\n'+
                                '  <A>를 누를 시 <Q>와 <W>를 얻은 좌표를 리스트에 추가합니다.\n'+
                                '  <E>를 누를 시 수를 예측합니다.\n'+
                                '  <M>을 눌러 무료 새로고침 버튼의 좌표 저장하세요.\n'
                                '  <N>을 눌러 생성기 프로그램의 빈 공간에 좌표를 저장하세요.\n\n'
                                '[사용법]\n'+
                                '  1. <Q>와 <W>를 눌러 사각형 좌표를 얻고 <A>를 눌러 리스트에 추가합니다.\n'+
                                '  2. (1)을 반복해서 여러 좌표를 얻을 수 있습니다.\n'+
                                '  3. <E>를 눌러 리스트에 저장된 좌표 쌍의 수만큼 캡처되고, 수를 예측합니다.\n'
                                '  4. 예측된 수가 UI에 출력됩니다.\n\n'
                                '[타이머 자동 사용법]\n'
                                ' 진행 : 마우스를 M의 좌표 이동 후 클릭 -> 예측(E와 같은 기능) -> '+
                                '마우스를 N의 좌표로 이동 후 클릭 -> 615초 후 이전 과정을 반복\n'
                                ' 1. [사용법] (1), (2)을 수행합니다.\n'
                                ' 2. N와 M의 좌표를 저장합니다.\n'
                                ' 3. [자동시작] 버튼을 눌러 데이터 자동생성을 시작하세요. [멈춤] 버튼을 제공합니다.\n\n'
                                '[자동 사용시 유의사항]\n'
                                ' - 숫자 데이터를 추출할 프로그램과 생성기 프로그램을 항상 모니터위에 보이도록 배치 해야합니다.\n'
                                ' - 두 프로그램이 서로 화면에 교차되더라도 마우스 N과 M 좌표는 서로 겹쳐지지 않게 배치 해야합니다.\n'
                                )
    #btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        self.close()
        
    def setText(self, msg):
        self.lb_event.clear()
        self.lb_event.setText(msg)
        
    def setUI(self):
        # 프레임과 레이아웃 선언
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0,450,150)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
    
        main_layout=QVBoxLayout()
        layout_top = QVBoxLayout()
        
        # 버튼 및 라벨 설정
        btn_start = QPushButton('자동시작', self)
        btn_stop = QPushButton('멈춤', self)
        
        btn_guide = QPushButton('사용법', self)       
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        btn_start.setGeometry(10, 10, 100 , 30)
        btn_stop.setGeometry(110, 10, 100, 30)
        self.lb_event.setGeometry(10, 70, 410, 25)
        btn_guide.setGeometry(210, 10, 100, 30)
        btn_quit.setGeometry(310, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_start)
        layout_top.addWidget(btn_stop)
        layout_top.addWidget(btn_guide)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        
        # 각 버튼에 콜백함수 연결
        btn_start.clicked.connect(self.threadFunction)
        btn_stop.clicked.connect(self.threadFunction)
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
        
app = QApplication(sys.argv)
win = DigitCapture()
win.show()
app.exec_()



