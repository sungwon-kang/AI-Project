import sys
import numpy as np

import pyautogui    # mouse, keyboard 관련 도구함 끌어오기
import pyperclip    # 클립보드에 저장 혹은 클립보드에서 불러오기 위함
import Imageprocessor as ip

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from tensorflow.keras.models import load_model

IP = ip.Imageprocessor()
#%%
class DigitCapture(QMainWindow):
    
    # P1 좌표 변수
    x1=0
    x2=0
    
    # P2 좌표 변수
    y1=0
    y2=0
    
    # P1, P2 각각 쌍으로 저장하는 리스트
    saved_p1_cord=[]
    saved_p2_cord=[]
    
    # 키 이벤트 락
    keyFlag=True
    
    captured_img=None       # 캡쳐 이미지 객체
    proceed_imgs=[]         # 분할된 이미지를 전처리된 이미지를 저장하는 이미지 리스트
    predicted_list=[]       # 예측된 수들을 저장하는 리스트
    
    def __init__(self):
        
        # 메인 윈도우 초기화
        super().__init__()
        self.setWindowTitle('Auto Arena[ver0.1]')
        self.setGeometry(200, 200, 600, 300)
        self.setFixedSize(600, 300)
        
        # UI 초기화
        self.setUI()
        
        # 학습 모델 불러오기
        self.cnn=load_model('./0_models/cnn_v5.h5')
        
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        # 키 입력 제한 리스트
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_A, Qt.Key_E, 
                   Qt.Key_C, Qt.Key_X]
        
        # 입력 키 가져오기
        key=e.key()
        if(self.keyFlag==True and key in printable):
            if key == Qt.Key_Q:
                # 마우스 포인터 좌표 불러오기
                self.x1, self.y1 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x1,self.y1))
                self.setText("위치 입력 (x1, y1) ({}, {})".format(self.x1,self.y1))
            
            elif key == Qt.Key_W:
                # 마우스 포인터 좌표 불러오기
                self.x2, self.y2 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x2,self.y2))
                self.setText("위치 입력 (x2, y2) ({}, {})".format(self.x2,self.y2))
                
            elif key == Qt.Key_A: 
                # 저장된 좌표 P1, P2 리스트에 추가
                self.saved_p1_cord.append((self.x1,self.y1))
                self.saved_p2_cord.append((self.x2,self.y2))
                
                print(self.saved_p1_cord)
                print(self.saved_p2_cord)
                self.setText("리스트 추가됨 "+str(self.saved_p1_cord) + str(self.saved_p2_cord))
            
            elif key == Qt.Key_C:
                # 좌표 리스트 초기화
                self.saved_p1_cord.clear()
                self.saved_p2_cord.clear()
                print("list clear")
                self.setText("리스트 초기화됨")
                
            elif key == Qt.Key_E:
                self.CaptureFunction()
                
            elif key == Qt.Key_X:
                self.quitFunction()
                    
    def CaptureFunction(self):
        # 모니터 스크린샷 가져오기
        l1=len(self.saved_p1_cord)
        l2=len(self.saved_p2_cord)       
        if( ( l1>0 and l2 > 0) and (l1 == l2) ):
            n = len(self.saved_p1_cord)
            self.keyFlag=False
            
            # 추가된 좌표 쌍 갯수 만큼 반복
            for i in range(n):
                (x1,y1)=self.saved_p1_cord[i]
                (x2,y2)=self.saved_p2_cord[i]
                print("두 위치 ({}, {}), ({}, {})".format(x1,y1,x2,y2))
                
                # 좌표 정렬 후 차이를 구하여 너비와 높이를 계산
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
        
                if 'x_re' in locals(): # 메모리 변수 속에 있는 것을 불러오기.
                    self.captured_img=pyautogui.screenshot(region=(sorted_x_value[0], sorted_y_value[0], width, height))
                    print("좌표대로 이미지 저장")
                    
                    # 전처리 함수 호출
                    self.processingFunction()
            
            self.setText("예측된 값 "+str(self.predicted_list))
            self.predicted_list.clear()
            
            # 끝나고 key 락 풀기
            self.keyFlag=True

    def resetFunction(self):
        self.le_cut.setText('1')
        self.le_crop_n.setText('7')
        self.le_fx1.setText('0')
        self.le_fx2.setText('0')
    
    def getParameterToLine(self):
        cut = int(self.le_cut.text())
        crop_n = int(self.le_crop_n.text())
        crop_fx1 = int(self.le_fx1.text())
        crop_fx2 = int(self.le_fx2.text())
        
        return cut, crop_n, crop_fx1, crop_fx2
    
    def processingFunction(self):
        # 분할 함수 호출
        dvi, n, fx1, fx2=self.getParameterToLine()
        cropped_imgs=IP.crop(cv_img=self.captured_img, cut=dvi, crop_n=n, crop_fx1=fx1, crop_fx2=fx2)
        print("이미지 분할 완료")
        
        # 각 분할된 이미지들을 전처리
        for crop_img in cropped_imgs:
            img = IP.preprocessing(crop_img)
            self.proceed_imgs.append(img)
        print("이미지 전처리 완료")
        
        # 예측 함수
        self.predictFunction()
        
    def predictFunction(self):
        
        if len(self.proceed_imgs)==0:
            self.setText("등록된 이미지가 없습니다.")
        else:
            # 처리된 이미지들을 모델 입력 데이터 형태로 변환
            proceed_imgs=IP.listToArray(self.proceed_imgs,'float32')
            res=self.cnn.predict(proceed_imgs)           
            
            # 각 자릿수를 계산
            num=0
            n_digit = len(res)-1
            for digit in res:
                # 예측된 확률 중 가장 큰 인덱스를 가져옴
                predicted_num=np.argmax(digit)
                
                # 자릿수 연결
                num+=predicted_num*(10**n_digit)
                n_digit=n_digit-1
                
                # 예측된 각 수를 출력
                print(predicted_num)
            
            # 예측된 수 추가
            self.predicted_list.append(num)
            self.proceed_imgs.clear()
            print("예측 값:", num)
        
        # 여기까지가 하나 이미지에 대한 예측 끝
        # 작업할 좌표 리스트이 남아있다면 103번 줄로 재진행
        
    # btn_guide(사용법) 이벤트 함수
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', 
                                '[단축키]\n'+
                                '  <Q>를 누를 시 마우스 포인터 기준으로 x1, y1이 저장됩니다.\n'+
                                '  <W>를 누를 시 마우스 포인터 기준으로 x2, y2이 저장됩니다.\n'+
                                '  <A>를 누를 시 <Q>와 <W>를 얻은 좌표를 리스트에 추가합니다.\n'+
                                '  <E>를 누를 시 수를 예측합니다.\n'+
                                '  <X>를 누를 시 프로그램이 종료됩니다.\n\n'+
                                
                                '[사용법]\n'+
                                '  1. <Q>와 <W>를 눌러 사각형 좌표를 얻고 <A>를 눌러 리스트에 추가합니다.\n'+
                                '  2. (1)을 반복해서 여러 좌표를 얻을 수 있습니다.\n'+
                                '  3. <E> 또는 [시작] 버튼을 눌러 리스트에 저장된 좌표 쌍의 수만큼 캡쳐되고, \t\t수를 예측합니다.\n'
                                '  4. 예측된 수가 UI에 출력됩니다.\n\n'
                                
                                '[설정]\n'+
                                ' 캡쳐 이미지 절단: 캡쳐 이미지를 얻어올 때, \n\t\t이미지를 1/n하여 얻어오는 설정값 (기본값 :1)\n'
                                ' 분할 이미지 왼쪽 너비 조절: 각 분할 이미지의 \n\t좌변 너비 크기 설정값 (기본값 :0)\n'
                                ' 분할 이미지 오른쪽 너비 조절: 각 분할 이미지의 \n\t우변 너비 크기 설정값 (기본값 :0)\n'
                                ' 분할 수: 캡쳐 이미지 분할 수 설정값 (기본값 :7)\n'
                                )
    
    # btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        self.close()
    
    def setText(self, msg):
        self.te_event.clear()
        self.te_event.setText(msg)
        
    def setUI(self):
        # 프레임과 레이아웃 선언
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0, 610, 147)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
        
        frame_bottom = QFrame(self)
        frame_bottom.setGeometry(0, 146, 610, 154)
        frame_bottom.setFrameShape(QFrame.Box | QFrame.Plain)
        
        main_layout=QVBoxLayout()
        layout_top =QVBoxLayout()
        layout_bottom=QVBoxLayout()
        
        # 상단 위젯
        lb_top = QLabel('설정', self)
        lb_top.setStyleSheet("font-weight: bold;"
                             "font-size: 20px"
                             )
        lb_top.setAlignment(Qt.AlignCenter)        
        
        lb_cut = QLabel('원본 이미지 절단', self)
        self.le_cut = QLineEdit(self)
        self.le_cut.setValidator(QIntValidator(2, 9, self))  # 1 ~ 10 사이의 정수 입력
        self.le_cut.setPlaceholderText("1")
        
        self.le_cut.setAlignment(Qt.AlignCenter)
        
        lb_crop_n = QLabel('분할 수', self)
        self.le_crop_n = QLineEdit(self)
        self.le_crop_n.setValidator(QIntValidator(self))# 1 ~ 10 사이의 정수 입력
        self.le_crop_n.setPlaceholderText("7")
        self.le_crop_n.setAlignment(Qt.AlignCenter)
        
        lb_fx1 = QLabel('분할 이미지 왼쪽 너비 간격', self)
        self.le_fx1 = QLineEdit(self)
        self.le_fx1.setPlaceholderText("0")
        self.le_fx1.setAlignment(Qt.AlignCenter)
        
        lb_fx2 = QLabel('분할 이미지 오른쪽 너비 간격', self)
        self.le_fx2 = QLineEdit(self)  
        self.le_fx2.setPlaceholderText("0")
        self.le_fx2.setAlignment(Qt.AlignCenter)
        
        # 초기 설정값
        self.resetFunction()
        
        lb_top.setGeometry(10,0,600,30)
        lb_cut.setGeometry(10,35,130,25)
        self.le_cut.setGeometry(10, 60, 130, 25)
        
        lb_crop_n.setGeometry(175, 35, 130, 25)
        self.le_crop_n.setGeometry(175, 60, 130, 25)
        
        lb_fx1.setGeometry(10, 85, 180, 25)
        self.le_fx1.setGeometry(10, 110, 130, 25)
    
        lb_fx2.setGeometry(175, 85, 180, 25)
        self.le_fx2.setGeometry(175, 110, 130, 25)
        
        layout_top.addWidget(lb_top)
        layout_top.addWidget(self.le_cut)
        layout_top.addWidget(self.le_crop_n)
        layout_top.addWidget(self.le_fx1)
        layout_top.addWidget(self.le_fx2)
        
        # 하단 위젯
        self.te_event = QTextEdit('사용법을 확인하세요!', self)
        self.te_event.setReadOnly(True)
        
        btn_start = QPushButton('시작(E)', self)
        btn_reset = QPushButton('설정 초기화',self)
        btn_guide = QPushButton('사용법', self)       
        btn_quit = QPushButton('나가기', self)
        
        self.te_event.setGeometry(10, 160, 580, 90)
        btn_start.setGeometry(10, 260, 100, 30)
        btn_reset.setGeometry(120, 260, 100, 30)
        btn_guide.setGeometry(380, 260, 100, 30)
        btn_quit.setGeometry(490, 260, 100, 30)
        
        layout_bottom.addWidget(self.te_event)
        layout_bottom.addWidget(btn_start)
        layout_bottom.addWidget(btn_reset)
        layout_bottom.addWidget(btn_guide)
        layout_bottom.addWidget(btn_quit)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
        # 각 버튼에 콜백함수 연결
        btn_start.clicked.connect(self.CaptureFunction)
        btn_reset.clicked.connect(self.resetFunction)
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
        self.te_event.setFocus()
        
def main():
    file=open('./ref/qt_man.css','r',encoding='utf-8')
    stylesheet=file.read()
    
    app = QApplication(sys.argv)
    win = DigitCapture()
    win.setStyleSheet(stylesheet)
    win.show()
    app.exec_()

main()

