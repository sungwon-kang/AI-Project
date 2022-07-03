import sys
import numpy as np

import pyautogui    # mouse, keyboard 관련 도구함 끌어오기
import pyperclip    # 클립보드에 저장 혹은 클립보드에서 불러오기 위함
import Imageprocessor_ver0_1 as ip

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from tensorflow.keras.models import load_model

# 키보드 제어 https://blankspace-dev.tistory.com/417
# 마우스 제어 https://codetorial.net/pyautogui/mouse_control.html

#%%
# 처리 순서
# 0. 좌표 입력
# 1. 전처리
# 2. 예측
# 3. 숫자연결
# 4. 출력
#%%
class DigitCapture(QMainWindow):

    x1 = 0
    y1 = 0
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('Auto Arena[ver0.1]')
        self.setGeometry(200, 200, 600, 100)
        self.setFixedSize(600, 100)
        
        self.lb_event = QLabel('사용법을 확인하세요!',self)
        self.setUI()
        
        
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_A, Qt.Key_E, Qt.Key_C]
        key=e.key()
        
        #press(key,presses=[], interval) 단일 키 입력, interval은 초 단위
        #typewrite는 여러 키를 입력할 수 있음.
        #
        if(key in printable):
            if key == Qt.Key_Q:
                self.x1, self.y1 = pyautogui.position()
                print("위치 값 ({}, {})\n".format(self.x1,self.y1))
                self.setText("위치 입력 (x1, y1) ({}, {})".format(self.x1,self.y1))
                
                # pyautogui.press("w")
                
            # elif key == Qt.Key_W:
                # pyautogui.typewrite("a")
                # print("w가 눌려짐")
                
            elif key == Qt.Key_A: 
                # 마우스 이동 및 클릭
                pyautogui.click(self.x1, self.y1) # interval가능
                           
    # btn_guide(사용법) 이벤트 함수
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', 
                                '[단축키]\n'+
                                '  <Q>를 누를 시 마우스 포인터 기준으로 x1, y1이 저장됩니다.\n'+
                                '  <W>를 누를 시 마우스 포인터 기준으로 x2, y2이 저장됩니다.\n'+
                                '  <A>를 누를 시 <Q>와 <W>를 얻은 좌표를 리스트에 추가합니다.\n'+
                                '  <E>를 누를 시 수를 예측합니다.\n\n'+
                                '[사용법]\n'+
                                '  1. <Q>와 <W>를 눌러 사각형 좌표를 얻고 <A>를 눌러 리스트에 추가합니다.\n'+
                                '  2. (1)을 반복해서 여러 좌표를 얻을 수 있습니다.\n'+
                                '  3. <E>를 눌러 리스트에 저장된 좌표 쌍의 수만큼 캡처되고, 수를 예측합니다.\n'
                                '  4. 예측된 수가 UI에 출력됩니다.'
                                )
    # btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        self.close()
        
    def setText(self, msg):
        self.lb_event.clear()
        self.lb_event.setText(msg)
        
    def setUI(self):
        # 프레임과 레이아웃 선언
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0,600, 100)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
    
        main_layout=QVBoxLayout()
        layout_top = QVBoxLayout()
        
        # 버튼 및 라벨 설정
        # btn_processing = QPushButton('분할 및 전처리', self)
        # btn_predict = QPushButton('예측', self)
        
        btn_guide = QPushButton('사용법', self)       
        btn_quit = QPushButton('나가기', self)

        # 각 위젯 위치와 크기 지정 
        self.lb_event.setGeometry(10, 70, 590, 25)
        btn_guide.setGeometry(10, 10, 100, 30)
        btn_quit.setGeometry(490, 10, 100, 30)
                  

        #상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_guide)
        layout_top.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        
        # 각 버튼에 콜백함수 연결
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
    
app = QApplication(sys.argv)
win = DigitCapture()
win.show()
app.exec_()



