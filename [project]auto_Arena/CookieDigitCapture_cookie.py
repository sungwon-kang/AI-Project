import sys
import numpy as np

import pyautogui    # mouse, keyboard 관련 도구함 끌어오기
import pyperclip    # 클립보드에 저장 혹은 클립보드에서 불러오기 위함
import Imageprocessor_cookie as ip

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from tensorflow.keras.models import load_model

IP = ip.Imageprocessor()
#%%
class DigitCapture(QMainWindow):
    
    # 좌표 변수
    x1=0
    y1=0

    x2=0
    y2=0

    x3=0
    y3=0
    
    x4=0
    y4=0
    
    # 탐색할 전투력 최대값
    # Max=0
    
    # 전투력 P1, P2 좌표 저장 리스트
    saved_p1_cord=[]
    saved_p2_cord=[]
    
    # 트로피 P3, P4 좌표 저장 리스트
    saved_p3_cord=[]
    saved_p4_cord=[]
    
    # 키 이벤트 락
    keyFlag=True
    
    WindowWidth=600
    WindowHeight=450
    
    def __init__(self):
        
        # 메인 윈도우 초기화
        super().__init__()
        self.setWindowTitle('Auto Arena[cookie]')
        self.setGeometry(200, 200, self.WindowWidth, self.WindowHeight)
        self.setFixedSize(self.WindowWidth, self.WindowHeight)
        
        # UI 초기화
        self.setUI()
        
        # 학습 모델 불러오기
        self.cnn=load_model('./0_models/cnn_v5_1.h5')
        
    # 키 이벤트 캡처 기능
    def keyPressEvent(self, e):
        # 키 입력 제한 리스트
        printable=[Qt.Key_Q, Qt.Key_W, Qt.Key_R, Qt.Key_T, 
                   Qt.Key_A, Qt.Key_E, Qt.Key_C, Qt.Key_X]
        
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
            
            elif key == Qt.Key_R:
                self.x3, self.y3 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x3,self.y3))
                self.setText("위치 입력 (x3, y3) ({}, {})".format(self.x3,self.y3))
                
            elif key == Qt.Key_T:
                self.x4, self.y4 = pyautogui.position()
                
                print("위치 값 ({}, {})\n".format(self.x4,self.y4))
                self.setText("위치 입력 (x4, y4) ({}, {})".format(self.x4,self.y4))
            
                
            elif key == Qt.Key_A: 
                # 저장된 좌표 P1, P2 리스트에 추가
                self.saved_p1_cord.append((self.x1,self.y1))
                self.saved_p2_cord.append((self.x2,self.y2))
                self.saved_p3_cord.append((self.x3,self.y3))
                self.saved_p4_cord.append((self.x4,self.y4))
                
                print('p1 list: ', self.saved_p1_cord)
                print('p2 list: ', self.saved_p2_cord)
                print('p3 list: ', self.saved_p3_cord)
                print('p4 list: ', self.saved_p4_cord)
                
                self.setText("리스트 추가됨 \np1 - "+str(self.saved_p1_cord) +"\np2 - "+ str(self.saved_p2_cord))
                self.te_event.append("\np3 - "+str(self.saved_p3_cord) +"\np4 - "+ str(self.saved_p4_cord))
            
            elif key == Qt.Key_C:
                # 좌표 리스트 초기화
                self.saved_p1_cord.clear()
                self.saved_p2_cord.clear()
                self.saved_p3_cord.clear()
                self.saved_p4_cord.clear()
                
                print("list clear")
                self.setText("리스트 초기화됨")
                
            elif key == Qt.Key_E:
                self.run()
                
            elif key == Qt.Key_X:
                self.quitFunction()
                
    def run(self):
        self.keyFlag=False
        
        # 캡쳐
        powerImgs=self.CaptureFunction(self.saved_p1_cord, self.saved_p2_cord)
        throphyImgs=self.CaptureFunction(self.saved_p3_cord, self.saved_p4_cord)
        
        # 분할
        cut, powerDigit, throphyDigit, fx1, fx2 = self.getParameterTopLine()
        # 리스트가 아닌 각 이미지를 전달해야함.
        # 리스트로 전달하게 만들기
        # cropped_powerImgs=IP.crop(cv_img=powerImgs, cut=cut, crop_n=powerDigit, crop_fx1=fx1, crop_fx2=fx2)
        # cropped_trophyImgs=IP.crop(cv_img=throphyImgs, cut=cut, crop_n=throphyDigit, crop_fx1=fx1, crop_fx2=fx2)
        
        # 전처리
        proceed_powerImgs=self.processImgs(cropped_powerImgs)
        proceed_trophyImgs=self.processImgs(cropped_trophyImgs)
        
        # 예측
        powerNums=self.predictFunction(proceed_powerImgs)
        throphyNums=self.predictFunction(proceed_trophyImgs)
        
        print(powerNums)
        print(throphyNums)
        
        self.keyFlag=True
        
    def CaptureFunction(self, l1, l2):
        # 스크린샷 좌표 가져오기 
        p1=len(l1)
        p2=len(l2)
        
        isSame = (p1==p2)
        if( isSame == True and l1!=0 ):
            n = len(l1)
            
            # 추가된 좌표 쌍 갯수 만큼 반복
            captured_imgs=[]
            for i in range(n):
                (x1,y1)=l1[i]
                (x2,y2)=l2[i]
                print("전투력 위치 ({}, {}), ({}, {})".format(x1,y1,x2,y2))
                
                X, Y, Width, Height = self.getCaptureCoord(x1,x2,y1,y2)
                print("영역 : ({}, {}, {}, {})\n".format(X, Y, Width, Height))
                
                captured_imgs.append(pyautogui.screenshot(region=(X, Y, Width, Height)))
            
            return captured_imgs
        
        
    def predictFunction(self, imglist):
        
        if len(imglist)==0:
            self.setText("등록된 이미지가 없습니다.")
        else:
            # 이미 전처리된 이미지를 파라미터로 받는다.
            # 처리된 이미지들을 모델 입력 데이터 형태로 변환
            predNumList=[]
            for i in range(len(imglist)):
                proceed_imgs=IP.listToArray(imglist[i],'float32')
            
                res=self.cnn.predict(proceed_imgs)
                
                pred_num=getNumber(res)
                print("예측 값:", pred_num)
                
                predNumList.append(pred_num)
            return predNumList
        
    def getNumber(self, predNums):
        # 각 자릿수를 계산
        result=0
        digit = len(predNums)-1
        for NUM in predNums:
            
            # 예측된 확률 중 가장 큰 인덱스를 가져옴
            Output=np.argmax(NUM)
            # 예측된 각 수를 출력
            print(Output)
            
            # 자릿수 연결
            result+=Output*(10**digit)
            digit=digit-1
        return result
    
    # btn_guide(사용법) 이벤트 함수
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', 
                                '[단축키]\n'+
                                '  <Q>를 누를 시 마우스 포인터 기준으로 x1, y1이 저장됩니다.\n'+
                                '  <W>를 누를 시 마우스 포인터 기준으로 x2, y2이 저장됩니다.\n'+
                                '  <A>를 누를 시 <Q>와 <W>를 얻은 좌표를 리스트에 추가합니다.\n'+
                                '  <P>를 누를 시 수를 예측합니다.\n'+
                                '  <X>를 누를 시 프로그램이 종료됩니다.\n\n'+
                                
                                '[사용법]\n'+
                                '  1. <Q>와 <W>를 눌러 사각형 좌표를 얻고 <A>를 눌러 리스트에 추가합니다.\n'+
                                '  2. (1)을 반복해서 여러 좌표를 얻을 수 있습니다.\n'+
                                '  3. <E> 또는 [시작] 버튼을 눌러 리스트에 저장된 좌표 쌍의 수만큼 캡쳐되고, 수를 예측합니다.\n'
                                '  4. 예측된 수가 UI에 출력됩니다.\n\n'
                                
                                '[설정]\n'+
                                ' 캡쳐 이미지 절단: 캡쳐 이미지를 얻어올 때, 이미지를 1/n 자르는 설정값\n(기본값 :1)\n'
                                ' 분할 이미지 왼쪽 너비 조절: 각 분할 이미지의 좌변 너비 크기 설정값\n(기본값 :0)\n'
                                ' 분할 이미지 오른쪽 너비 조절: 각 분할 이미지의 우변 너비 크기 설정값\n(기본값 :0)\n'
                                ' 분할 수: 캡쳐 이미지 분할 수 설정값\n(기본값 :7)\n'
                                )
    def processImgs(self, imgs):
        proceed_imgs=[]
        for i in range(len(imgs)):
            img=imgs[i]
            procced_img = IP.preprocessing(img)
            proceed_imgs.append(procced_img)
            
        return proceed_imgs
    
    def resetFunction(self):
        self.le_cut.setText('1')
        self.le_cropPower_n.setText('7')
        self.le_cropthrophy_n.setText('4')
        self.le_fx1.setText('0')
        self.le_fx2.setText('0')
        
    # btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        self.close()
    
    def printInList(self):
        # 
        # Max, throphy = self.getParameterMiddleLine()
        # lst = [i for i in lst if i <= self.Max]
        self.te_event.append("예측된 수: "+str(self.predicted_list))
        self.te_event.append("\n나보다 낮은 전투력들: "+str(self.findInList(self.predicted_list)))
        
    def getCaptureCoord(self, x1, x2, y1,y2):
        # 좌표 정렬 후 차이를 구하여 너비와 높이를 계산
        sorted_X_value = sorted([x1, x2])
        sorted_Y_value = sorted([y1, y2])
        Width = sorted_X_value[1] - sorted_X_value[0]
        Height = sorted_Y_value[1] - sorted_Y_value[0]
                
        return sorted_X_value[0], sorted_Y_value[0], Width, Height
    
    def getParameterTopLine(self):
        cut = int(self.le_cut.text())
        powerDigit = int(self.le_cropPower_n.text())
        throphyDigit = int(self.le_cropthrophy_n.text())
        crop_fx1 = int(self.le_fx1.text())
        crop_fx2 = int(self.le_fx2.text())
        
        return cut, powerDigit, throphyDigit, crop_fx1, crop_fx2
    
    def getParameterMiddleLine(self):
        throphy = int(self.le_EnemyMaxtrophy.text())
        Max = int(self.le_EnemyMaxPower.text())
        
        return Max, throphy

    def setText(self, msg):
        self.te_event.clear()
        self.te_event.setText(msg)
        
    def setUI(self):
        # 프레임과 레이아웃 선언
        frame_top = QFrame(self)
        frame_top.setGeometry(0, 0, self.WindowWidth, 150)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
        
        frame_bottom = QFrame(self)
        frame_bottom.setGeometry(0, 149, self.WindowWidth, 150)
        frame_bottom.setFrameShape(QFrame.Box | QFrame.Plain)
        
        main_layout=QVBoxLayout()
        layout_top =QVBoxLayout()
        layout_middle =QVBoxLayout()
        layout_bottom=QVBoxLayout()
        
        # 상단 위젯
        lb_top = QLabel('기본 설정', self)
        lb_top.setStyleSheet("font-weight: bold;"
                             "font-size: 20px"
                             )
        lb_top.setAlignment(Qt.AlignCenter)        
        
        lb_cut = QLabel('원본 이미지 절단', self)
        self.le_cut = QLineEdit(self)
        self.le_cut.setValidator(QIntValidator(1, 10, self))# 1 ~ 10 사이의 정수 입력
        self.le_cut.setPlaceholderText("1")
        
        self.le_cut.setAlignment(Qt.AlignCenter)
        
        lb_cropPower_n = QLabel('전투력 분할 수', self)
        self.le_cropPower_n = QLineEdit(self)
        self.le_cropPower_n.setValidator(QIntValidator(self))
        self.le_cropPower_n.setPlaceholderText("7")
        self.le_cropPower_n.setAlignment(Qt.AlignCenter)
        
        lb_cropthrophy_n = QLabel('트로피 분할 수', self)
        self.le_cropthrophy_n = QLineEdit(self)
        self.le_cropthrophy_n.setValidator(QIntValidator(self))
        self.le_cropthrophy_n.setPlaceholderText("4")
        self.le_cropthrophy_n.setAlignment(Qt.AlignCenter)
        
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
        
        lb_top.setGeometry(0,0, self.WindowWidth, 30)
        
        lb_cut.setGeometry(10,35,130,25)
        self.le_cut.setGeometry(10, 60, 130, 25)
        
        lb_cropPower_n.setGeometry(175, 35, 130, 25)
        self.le_cropPower_n.setGeometry(175, 60, 130, 25)
        
        lb_cropthrophy_n.setGeometry(335, 35, 130, 25)
        self.le_cropthrophy_n.setGeometry(335, 60, 130, 25)
        
        lb_fx1.setGeometry(10, 85, 180, 25)
        self.le_fx1.setGeometry(10, 110, 130, 25)
    
        lb_fx2.setGeometry(175, 85, 180, 25)
        self.le_fx2.setGeometry(175, 110, 130, 25)
        
        layout_top.addWidget(lb_top)
        layout_top.addWidget(lb_cut)
        layout_top.addWidget(lb_cropPower_n)
        layout_top.addWidget(lb_cropthrophy_n)
        layout_top.addWidget(lb_fx1)
        layout_top.addWidget(lb_fx2)
        layout_top.addWidget(self.le_cut)
        layout_top.addWidget(self.le_cropPower_n)
        layout_top.addWidget(self.le_cropthrophy_n)
        layout_top.addWidget(self.le_fx1)
        layout_top.addWidget(self.le_fx2)
        
        # 중단 위젯
        lb_middle = QLabel('쿠키런', self)
        lb_middle.setStyleSheet("font-weight: bold;"
                             "font-size: 20px")
        lb_middle.setAlignment(Qt.AlignCenter)
        
        lb_EnemyMaxtrophy = QLabel('탐색할 상대 트로피 최대치', self)
        self.le_EnemyMaxtrophy = QLineEdit(self)
        self.le_EnemyMaxtrophy.setValidator(QIntValidator(self))
        self.le_EnemyMaxtrophy.setAlignment(Qt.AlignCenter)
        self.le_EnemyMaxtrophy.setText('0')
        
        lb_EnemyMaxPower = QLabel('탐색할 상대 전투력 최대치', self)
        self.le_EnemyMaxPower = QLineEdit(self)
        self.le_EnemyMaxPower.setValidator(QIntValidator(self))
        self.le_EnemyMaxPower.setAlignment(Qt.AlignCenter)
        self.le_EnemyMaxPower.setText('0')
        
        lb_middle.setGeometry(0,149,self.WindowWidth, 30)
        
        lb_EnemyMaxtrophy.setGeometry(10,185,150,25)
        self.le_EnemyMaxtrophy.setGeometry(10,210,130,25)
        
        lb_EnemyMaxPower.setGeometry(175,185,150,25)
        self.le_EnemyMaxPower.setGeometry(175,210,130,25)

        layout_top.addWidget(lb_middle)
        layout_middle.addWidget(lb_middle)
        layout_middle.addWidget(self.le_EnemyMaxtrophy)
        layout_middle.addWidget(lb_EnemyMaxPower)
        layout_middle.addWidget(self.le_EnemyMaxPower)

        
        # 하단 위젯
        self.te_event = QTextEdit('사용법을 확인하세요!', self)
        self.te_event.setReadOnly(True)
        
        btn_start = QPushButton('시작(E)', self)
        btn_reset = QPushButton('설정 초기화',self)
        btn_guide = QPushButton('사용법', self)       
        btn_quit = QPushButton('종료', self)
        
        self.te_event.setGeometry(10, self.WindowHeight-140, 580, 90)
        btn_start.setGeometry(10, self.WindowHeight-40, 100, 30)
        btn_reset.setGeometry(120, self.WindowHeight-40, 100, 30)
        btn_guide.setGeometry(380, self.WindowHeight-40, 100, 30)
        btn_quit.setGeometry(490, self.WindowHeight-40, 100, 30)
        
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