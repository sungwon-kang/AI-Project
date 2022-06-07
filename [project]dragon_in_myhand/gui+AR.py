import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import winsound
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#%%

# MediaPipe를 이용한 손 인식 이미지 합성 프로그램
class InMyHand(QMainWindow):
    
    # 손 이미지 크기 변수
    img_w=None
    img_h=None

    # 효과 이미지 크기 변수
    e_w=None
    e_h=None
    alpha=None
    
    # 각 이미지 저장 변수
    handImg=None
    effectImg=None
    composedImg=None
    
    # mediapipe hands 객체
    wrist=None
    
    # x, y 위치 변수
    x=0
    y=0
    
    # 손 이미지 크기 배율 변수
    img_fx=0.1
    img_fy=0.1
    
    # 효과 이미지 크기 배율 변수
    effect_fx=0.2
    effect_fy=0.2
    
    def __init__(self):
        
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        self.setGeometry(200, 200, 800, 250)
        self.setFixedSize(800, 250)
        
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
        
        self.msg_label = QLabel('환영합니다!', self)
        self.lb_x = QLabel('위치이동 x '+str(self.x), self)
        self.lb_y = QLabel('위치이동 y '+str(self.y), self)
        
        self.lb_handimg_fx = QLabel('손 이미지 너비 배율 '+str(self.img_fx), self)
        self.lb_handimg_fy = QLabel('손 이미지 높이 배율 '+str(self.img_fy), self)
        
        self.lb_effectimg_fx = QLabel('효과 이미지 너비 배율 '+str(self.effect_fx), self)
        self.lb_effectimg_fy = QLabel('효과 이미지 높이 배율 '+str(self.effect_fy), self)
        
        # 각 슬라이드 설정
        x_slider = QSlider(Qt.Horizontal, self)
        x_slider.adjustSize()
        x_slider.setTickInterval(1)
        x_slider.setRange(-300, 300)
        x_slider.setSingleStep(1)
        
        y_slider = QSlider(Qt.Horizontal, self)
        y_slider.adjustSize()
        y_slider.setTickInterval(1)
        y_slider.setRange(-300, 300)
        y_slider.setSingleStep(1)

        hand_x_slider = QSlider(Qt.Horizontal, self)
        hand_x_slider.adjustSize()
        hand_x_slider.setTickInterval(1)
        hand_x_slider.setRange(0, 10)
        hand_x_slider.setSingleStep(1)
        
        hand_y_slider = QSlider(Qt.Horizontal, self)
        hand_y_slider.adjustSize()
        hand_y_slider.setTickInterval(1)
        hand_y_slider.setRange(0, 10)
        hand_y_slider.setSingleStep(1)
        
        effect_x_slider = QSlider(Qt.Horizontal, self)
        effect_x_slider.adjustSize()
        effect_x_slider.setTickInterval(1)
        effect_x_slider.setRange(0, 10)
        effect_x_slider.setSingleStep(1)
        
        effect_y_slider = QSlider(Qt.Horizontal, self)
        effect_y_slider.adjustSize()
        effect_y_slider.setTickInterval(1)
        effect_y_slider.setRange(0, 10)
        effect_y_slider.setSingleStep(1)
        
        # 각 위젯 위치와 크기 지정 
        btn_loadhand.setGeometry(10, 10, 100, 30)
        btn_loadeffect.setGeometry(110, 10, 100 , 30)
        btn_recognition.setGeometry(210, 10, 100, 30)
        btn_compose.setGeometry(310, 10, 100, 30)
    
        btn_guide.setGeometry(490, 10, 100, 30)
        btn_save.setGeometry(590, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)

        x_slider.setGeometry(30, 110, 120, 30)
        self.lb_x.setGeometry(180, 75, 140, 100)
        
        y_slider.setGeometry(370, 110, 120, 30)
        self.lb_y.setGeometry(520, 75, 140, 100)
        
        hand_x_slider.setGeometry(30, 150, 120, 30)
        self.lb_handimg_fx.setGeometry(180, 115, 140, 100)
        
        hand_y_slider.setGeometry(370, 150, 120, 30)
        self.lb_handimg_fy.setGeometry(520, 115, 140, 100)
        
        effect_x_slider.setGeometry(30, 190, 120, 30)
        self.lb_effectimg_fx.setGeometry(180, 155, 140, 100)
        
        effect_y_slider.setGeometry(370, 190, 120, 30)
        self.lb_effectimg_fy.setGeometry(520, 155, 140, 100)
        
        
        self.msg_label.setGeometry(10, 65, 600, 50)
        
        # 상단 프레임 레이아웃 위젯 추가
        layout_top.addWidget(btn_loadhand)
        layout_top.addWidget(btn_loadeffect)
        layout_top.addWidget(btn_recognition)
        layout_top.addWidget(btn_compose)
        layout_top.addWidget(btn_guide)
        layout_top.addWidget(btn_save)
        layout_top.addWidget(btn_quit)
        layout_top.addWidget(self.msg_label)
        
        # 하단 프레임 레이아웃 위젯 추가
        layout_bottom.addWidget(x_slider)
        layout_bottom.addWidget(y_slider)
        layout_bottom.addWidget(self.lb_x)
        layout_bottom.addWidget(self.lb_y)
        
        layout_bottom.addWidget(hand_x_slider)
        layout_bottom.addWidget(hand_y_slider)
        layout_bottom.addWidget(self.lb_handimg_fx)
        layout_bottom.addWidget(self.lb_handimg_fy)
        
        layout_bottom.addWidget(effect_x_slider)
        layout_bottom.addWidget(effect_y_slider)
        layout_bottom.addWidget(self.lb_effectimg_fx)
        layout_bottom.addWidget(self.lb_effectimg_fy)
        
        
        # 각 버튼에 콜백함수 연결
        btn_loadhand.clicked.connect(self.loadhandfunction)
        btn_loadeffect.clicked.connect(self.loadeffectFunction)
        btn_recognition.clicked.connect(self.recognitionFunction)
        btn_compose.clicked.connect(self.composeFunction)
        btn_save.clicked.connect(self.saveImgFunction)
        btn_guide.clicked.connect(self.showGuideFunction)
        btn_quit.clicked.connect(self.quitFunction)
        
        # 각 슬라이드에 콜백함수 연결
        x_slider.valueChanged.connect(self.x_value_changed)
        y_slider.valueChanged.connect(self.y_value_changed)  
        hand_x_slider.valueChanged.connect(self.img_fx_changed)
        hand_y_slider.valueChanged.connect(self.img_fy_changed)
        effect_x_slider.valueChanged.connect(self.effect_fx_changed)
        effect_y_slider.valueChanged.connect(self.effect_fy_changed)
        
        # 메인 레이아웃에 모든 프레임을 추가
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
    # btn_guide(사용법) 이벤트 함수    
    def showGuideFunction(self):
        QMessageBox.information(self, '사용법', '\n[사용 순서]\n1. 손 불러오기 버튼을 눌러 손 이미지를 선택합니다.\n'+
                                '2. 효과 불러오기 버튼을 눌러 효과 이미지를 선택합니다.\n'+
                                '3. 손 인식하기 버튼을 눌러 손 이미지에서 손을 찾습니다.\n'+
                                '4. 이미지 합성 버튼을 눌러 손 이미지와 효과 이미지를 합성합니다.\n'+
                                
                                '\n[설정]\n'
                                'tip1. 손 이미지 배율 슬라이더를 이용하면 불러올 손 이미지 크기 배율을 조절할 수 있습니다.\n'
                                'tip2. 효과 이미지 배율 슬라이더를 이용하여 불러올 효과 이미지 크기 배율을 조절할 수 있습니다.\n'
                                'tip3. (x,y)위치이동 슬라이더를 이용하여 합성된 이미지에서 효과 부분을 옮길 수 있습니다.\n'                            
                                
                                '\n[제공]\n'
                                '  Sample폴더에 제 손 이미지와 기본 3개 효과이미지를 제공합니다.\n프로그램이 정상적으로 실행되길 기대합니다!'
                                )
    
    # x, y 이동 슬라이드 이벤트 함수
    def x_value_changed(self,value):  
        self.x=int(value)
        self.lb_x.setText('효과 위치이동 x '+str(self.x))
        print(self.x)
        
    def y_value_changed(self,value):  
        self.y=int(value)
        self.lb_y.setText('효과 위치 이동 '+str(self.y))
        print(self.y)
        
    # 손 이미지 크기 배율 슬라이드 이벤트 함수
    def img_fx_changed(self,value):  
        self.img_fx=float(value)/10
        self.lb_handimg_fx.setText('손 이미지 너비 배율 '+str(self.img_fx))
        print(self.img_fx)
        
    def img_fy_changed(self,value):  
        self.img_fy=float(value)/10
        self.lb_handimg_fy.setText('손 이미지 높이 배율 '+str(self.img_fy))
        print(self.img_fy)
        
    # 효과 이미지 크기 배율 슬라이드 이벤트 함수
    def effect_fx_changed(self,value):  
        self.effect_fx=float(value)/10
        self.lb_effectimg_fx.setText('효과 이미지 너비 배율 '+str(self.effect_fx))
        print(self.effect_fx)
        
    def effect_fy_changed(self,value):  
        self.effect_fy=float(value)/10
        self.lb_effectimg_fy.setText('효과 이미지 높이 배율 '+str(self.effect_fy))
        print(self.effect_fy)
        
        
    #btn_loadhand[손 불러오기] 이벤트 함수  
    def loadhandfunction(self):
        self.msg_label.clear()
    
         # 다이알로그로 파일의 경로를 가져옴
        handImg = QFileDialog.getOpenFileName(self,'','./')
        # 경로에 있는 이미지 읽어옴
        self.handImg = cv.imread(handImg[0])
        
        if self.handImg is not None:
            self.msg_label.setText('이미지를 불러왔습니다.')
            
            # 이미지 크기 재조정
            self.handImg = cv.resize(self.handImg, (0, 0), fx=self.img_fx, fy=self.img_fy, interpolation=cv.INTER_AREA)
                    
            # 이미지 크기 전역 변수에 저장
            self.img_w, self.img_h = self.handImg.shape[1], self.handImg.shape[0]
            
            # 저장된 배열 영상을 출력
            cv.imshow('My Hand', self.handImg)
        else:
            self.msg_label.setText('손 이미지를 읽지 못했습니다.')
       
    #btn_loadeffect[효과 불러오기] 이벤트 함수
    def loadeffectFunction(self):
        self.msg_label.clear()
       
        # 다이알로그로 파일의 경로를 가져옴
        effectImg=QFileDialog.getOpenFileName(self,'','./')
        
        # 경로에 있는 이미지 읽어옴
        self.effectImg = cv.imread(effectImg[0],cv.IMREAD_UNCHANGED)

        if self.effectImg is not None:
            self.msg_label.setText('효과를 불러왔습니다')
                                   
            # 이미지 크기 재조정
            self.effectImg=cv.resize(self.effectImg,(0,0),fx=self.effect_fx,fy=self.effect_fy,interpolation=cv.INTER_AREA)
        
            # 이미지 크기 전역 변수에 저장
            self.e_w, self.e_h = self.effectImg.shape[1], self.effectImg.shape[0]
            self.alpha=self.effectImg[:,:,3:]/255
            
            # 저장된 배열 영상을 출력
            cv.imshow('Effect', self.effectImg)
        else:
            self.msg_label.setText('효과 이미지를 읽지 못했습니다.')
        
    #btn_recognition[손 인식하기] 이벤트 함수  
    def recognitionFunction(self):    
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:
            
            # 손 인식하는 부분
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = hands.process(cv.cvtColor(self.handImg, cv.COLOR_BGR2RGB))

            # 사진에 찍힌 손이 왼손, 오른손을 판별한 결과를 출력
            print('Handedness:', results.multi_handedness)
            self.msg_label.clear()
            if not results.multi_hand_landmarks:
                self.msg_label.setText('손을 찾지 못했습니다.')
            
            else:
                self.msg_label.setText('손이 인식되었습니다')
                self.wrist=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]  
                
                meshimg=self.handImg.copy()
                
                # mesh를 그리는 부분
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        meshimg,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                # mesh가 그려진 이미지를 저장
                cv.imshow('Mesh Hand', meshimg)
            
           
            
    #btn_compose[이미지 합성] 이벤트 함수        
    def composeFunction(self):
        # 필요한 이미지와 인식 정보가 있다면 합성 실행
        if self.handImg is not None and self.effectImg is not None and self.wrist is not None:
            # 손 이미지 복사
            self.composedImg = self.handImg.copy()
                
            # 정규화된 손목 위치를 손 이미지 크기를 곱하여 손 이미지에서 손목의 위치를 얻는다.
            # 손목 위치에서 효과 이미지 너비와 높이를 계산하여 중앙이 손목 위치인 사각형 좌표(x1,x2,y1,y2)를 얻는다.
            # 사각형 좌표는 x, y 값으로 조절 가능하다.
            x1,x2=int(self.wrist.x*self.img_w-self.e_w/2)-self.x,int(self.wrist.x*self.img_w+self.e_w/2)-self.x
            y1,y2=int(self.wrist.y*self.img_h-self.e_h/2)-self.y,int(self.wrist.y*self.img_h+self.e_h/2)-self.y
            print(x1, x2)
            print(y1, y2)
            
            # 복사한 손 이미지에 사각형 좌표에 효과 이미지를 덮어 쓴다.
            if x1>0 and y1>0 and x2<self.img_w and y2<self.img_h:
                self.composedImg[y1:y2,x1:x2]=self.handImg[y1:y2,x1:x2]*(1-self.alpha)+self.effectImg[:,:,:3]*self.alpha
            
            cv.imshow('Composed Hand', self.composedImg)
            self.msg_label.setText('이미지가 합성되었습니다. 만약 효과가 보이지 않는다면 위치 이동을 다시 설정하세요!')
            
        else:
            self.msg_label.setText('우선 이미지 등록과 인식을 해주세요!')
        
        
    #btn_save[이미지 저장] 이벤트 함수
    def saveImgFunction(self):
        if self.composedImg is not None:
            cv.imwrite('ComposedImg', self.composedImg,1)           # 이미지 저장 함수
            self.msg_label.setText('합성된 이미지를 저장하였습니다.')
        else:
            self.msg_label.setText('저장할 이미지가 없습니다.')
    
    #btn_quit[나가기] 이벤트 함수    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = InMyHand()
win.show()
app.exec_()