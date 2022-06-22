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
        
	[전처리 기능]
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
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        
        self.setGeometry(200, 200, 800, 250)
        self.setFixedSize(800, 250)
        self.setWidget()
    
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_E]
        key=e.key()
        
        if(self.keyFlag==True and key in printable):
            if key == Qt.Key_Q:
                self.x1, self.y1 = pyautogui.position()
                print("위치 값 : ({}, {})\n".format(self.x1,self.y1))
            
            elif key == Qt.Key_W:
                self.x2, self.y2 = pyautogui.position()
                print("위치 값 : ({}, {})\n".format(self.x2,self.y2))
               
                
            elif key == Qt.Key_E:
                print("두 위치 : ({}, {}), ({}, {})".format(self.x1,self.y1,self.x2,self.y2))
                print("위치 값 : ({}, {})\n".format(self.x1,self.y1))

                sorted_x_value = sorted([self.x1, self.x2])
                sorted_y_value = sorted([self.y1, self.y2])
                width = sorted_x_value[1] - sorted_x_value[0]
                height = sorted_y_value[1] - sorted_y_value[0]

                x_re = sorted_x_value[0] - 20
                y_re = sorted_y_value[0] - 20
                width_re = width + 40
                height_re = height + 40
                                                  
                print("영역 : ({}, {}, {}, {})\n".format(x_re, y_re, width_re, height_re))
                pyperclip.copy("({}, {}, {}, {})".format(x_re, y_re, width_re, height_re))
                
                if 'x_re' in locals(): # 메모리 변수 속에 있는 것을 불러오기. 따라서 x_re이 있어야 함.
                    fileName = str("test") + ".jpg"
                    pyautogui.screenshot(fileName, region=(sorted_x_value[0], sorted_y_value[0], width, height))
                    print("저장완료")
                    
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
        btn_capture = QPushButton('temp', self)
        # btn_loadeffect = QPushButton('2. 효과 불러오기', self)
        # btn_recognition = QPushButton('3. 손 인식하기', self)
        # btn_compose = QPushButton('4. 이미지 합성', self)
        
        btn_guide = QPushButton('사용법', self)
        # btn_save = QPushButton('이미지 저장', self)        
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        btn_capture.setGeometry(10, 10, 100, 30)
        # btn_loadeffect.setGeometry(110, 10, 100 , 30)
        # btn_recognition.setGeometry(210, 10, 100, 30)
        # btn_compose.setGeometry(310, 10, 100, 30)
        
        btn_guide.setGeometry(490, 10, 100, 30)
        # btn_save.setGeometry(590, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_capture)
        # layout_top.addWidget(btn_loadeffect)
        # layout_top.addWidget(btn_recognition)
        # layout_top.addWidget(btn_compose)
        layout_top.addWidget(btn_guide)
        # layout_top.addWidget(btn_save)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
        # 각 버튼에 콜백함수 연결
        # btn_capture.clicked.connect(self.capturefunction)
        # btn_loadeffect.clicked.connect(self.loadeffectFunction)
        # btn_recognition.clicked.connect(self.recognitionFunction)
        # btn_compose.clicked.connect(self.composeFunction)
        # btn_save.clicked.connect(self.saveImgFunction)
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
    
app = QApplication(sys.argv)
win = DigitCapture()
win.show()
app.exec_()

#%%
# 성능 측정해보기
cnn=load_model('./models/cnn_v4.h5')

# cookie 훈련 집합과 검증 집합
IP=ip.Imageprocessor()
x_train_cookie=np.array(IP.load_imgs('trainSample',True), dtype='float32')
x_test_cookie=np.array(IP.load_imgs('valSample', False), dtype='float32')

# 부류를 원핫코드로 변환
y_train_cookie=np.array([0,1,2,3,4,5,6,7,8,9,0])
y_test_cookie=np.array([0,1,2,3,4,5,6,7,8,9,0])

y_train_cookie=tf.keras.utils.to_categorical(y_train_cookie,10)
y_test_cookie=tf.keras.utils.to_categorical(y_test_cookie,10)

# 배열 합치기
x_test=np.concatenate([x_train_cookie,x_test_cookie])
y_test=np.concatenate([y_train_cookie,y_test_cookie])

#%%
res=cnn.predict(x_test_cookie)
# 원 핫코드 디코딩
y_test = np.argmax(y_test_cookie, axis=1).reshape(-1,1)
#%%

for img in x_test:
    IP.show(img)
print(y_test_cookie)


# 혼동 행렬 구함 ( 예측 i , 실제 j )
conf=np.zeros((10,10))          #10x10 0으로 채운 행렬 생
for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
    conf[np.argmax(res[i])][y_test[i]]+=1 	 #res[i]측정한 값, y_test[i]실제 값 위치에 +1
    
print(conf)		# 출력, 대각선 부분이 예측과 실제값이 일치한 부분이다.

# 정확률 측정하고 출력
no_correct =0
for i in range(10):
    no_correct+=conf[i][i] # 혼동행렬의 대각선 부분을 모두 더한다.

accuracy = no_correct/len(res) # 모두 더한 값에 예측값 수을 나누면 정확도를 구할 수 있다.
print(accuracy*100,"%")


