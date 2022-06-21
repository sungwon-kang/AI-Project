<<<<<<< HEAD
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import winsound

#%%
# 1. PC 화면 일부분 캡처 기능 구현
# 2. 캡처한 사진 일부분 자르기(우선 하나만)
#%%
# MediaPipe를 이용한 손 인식 이미지 합성 프로그램
class AutoArena(QMainWindow):
    
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        
        self.setGeometry(200, 200, 800, 250)
        self.setFixedSize(800, 250)
        self.setWidget()
        
    
    # btn_guide(사용법) 이벤트 함수    
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', '')


    #btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        cv.destroyAllWindows()
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
        btn_loadhand = QPushButton('1. 손 불러오기', self)
        btn_loadeffect = QPushButton('2. 효과 불러오기', self)
        btn_recognition = QPushButton('3. 손 인식하기', self)
        btn_compose = QPushButton('4. 이미지 합성', self)
        
        btn_guide = QPushButton('사용법', self)
        btn_save = QPushButton('이미지 저장', self)        
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        btn_loadhand.setGeometry(10, 10, 100, 30)
        btn_loadeffect.setGeometry(110, 10, 100 , 30)
        btn_recognition.setGeometry(210, 10, 100, 30)
        btn_compose.setGeometry(310, 10, 100, 30)
        
        btn_guide.setGeometry(490, 10, 100, 30)
        btn_save.setGeometry(590, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_loadhand)
        layout_top.addWidget(btn_loadeffect)
        layout_top.addWidget(btn_recognition)
        layout_top.addWidget(btn_compose)
        layout_top.addWidget(btn_guide)
        layout_top.addWidget(btn_save)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)

app = QApplication(sys.argv)
win = AutoArena()
win.show()
=======
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import winsound

#%%
# 1. PC 화면 일부분 캡처 기능 구현
# 2. 캡처한 사진 일부분 자르기(우선 하나만)
#%%
# MediaPipe를 이용한 손 인식 이미지 합성 프로그램
class AutoArena(QMainWindow):
    
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        
        self.setGeometry(200, 200, 800, 250)
        self.setFixedSize(800, 250)
        self.setWidget()
        
    
    # btn_guide(사용법) 이벤트 함수    
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', '')


    #btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        cv.destroyAllWindows()
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
        btn_loadhand = QPushButton('1. 손 불러오기', self)
        btn_loadeffect = QPushButton('2. 효과 불러오기', self)
        btn_recognition = QPushButton('3. 손 인식하기', self)
        btn_compose = QPushButton('4. 이미지 합성', self)
        
        btn_guide = QPushButton('사용법', self)
        btn_save = QPushButton('이미지 저장', self)        
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        btn_loadhand.setGeometry(10, 10, 100, 30)
        btn_loadeffect.setGeometry(110, 10, 100 , 30)
        btn_recognition.setGeometry(210, 10, 100, 30)
        btn_compose.setGeometry(310, 10, 100, 30)
        
        btn_guide.setGeometry(490, 10, 100, 30)
        btn_save.setGeometry(590, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_loadhand)
        layout_top.addWidget(btn_loadeffect)
        layout_top.addWidget(btn_recognition)
        layout_top.addWidget(btn_compose)
        layout_top.addWidget(btn_guide)
        layout_top.addWidget(btn_save)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)

app = QApplication(sys.argv)
win = AutoArena()
win.show()
>>>>>>> a7a001ef1b0b37e7a9257206d72fe44574799c1c
app.exec_()