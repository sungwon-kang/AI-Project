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
# UI 구성 ok
# 다이알로그 이미지 불러오기 ok
# 불러온 이미지 하단 프레임에 출력하기 
# 손 인식하기 기능 넣기 + 인식이 안될 시 예외처리

# 슬라이더 바로 효과 이미지 수동 조정할 수 있게 하기 
    # - 진행상황
            # - 슬라이더 바 생성 ok
            # - 값 받기, 출력하기 
# 이미지 합성 기능 넣기 ok
# 사용법 버튼 만들기 + 내용 채우기



#%%


class InMyHand(QMainWindow):
    
    img_w=None
    img_h=None

    e_w=None
    e_h=None
    
    alpha=None
    
    handImg=None
    effectImg=None
    
    wrist=None
    
    y=50
    x=50
    
    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        
        main_layout=QVBoxLayout()
        
        self.setWindowTitle('내 손안에 흑염룡이')
        self.setGeometry(200, 200, 800, 500)
        
        # 버튼와 레이블 선언
        
        
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0,800,100)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
        
        frame_bottom = QFrame(self)
        frame_bottom.setGeometry(0,100,800,400)
        frame_bottom.setFrameShape(QFrame.Box | QFrame.Plain)
        
        layout_top = QVBoxLayout()
        layout_bottom = QVBoxLayout()
        
        btn_loadhand = QPushButton('손 불러오기', self)
        btn_loadeffect = QPushButton('효과 불러오기', self)
        btn_recognition = QPushButton('손 인식하기', self)
        btn_composeButton = QPushButton('이미지 합성', self)
        btn_quit = QPushButton('나가기', self)
        
        self.msg_label = QLabel('환영합니다!', self)
        self.x_label = QLabel('위치이동 x '+str(self.x), self)
        self.y_label = QLabel('위치이동 y '+str(self.y), self)
        
        x_slider = QSlider(Qt.Horizontal, self)
        x_slider.adjustSize()
        x_slider.setTickInterval(1)
        x_slider.setRange(-100, 100)
        x_slider.setSingleStep(1)
        
        y_slider = QSlider(Qt.Horizontal, self)
        y_slider.adjustSize()
        y_slider.setTickInterval(1)
        y_slider.setRange(-100, 100)
        y_slider.setSingleStep(1)
        
        btn_loadhand.setGeometry(10, 10, 100, 30)
        btn_loadeffect.setGeometry(110, 10, 100 , 30)
        btn_recognition.setGeometry(210, 10, 100, 30)
        btn_composeButton.setGeometry(310, 10, 100, 30)
        btn_quit.setGeometry(690, 10, 100, 30)
        
        x_slider.setGeometry(10, 50, 150, 30)
        self.x_label.setGeometry(180, 20, 100, 100)
        
        y_slider.setGeometry(300, 50, 150, 30)
        self.y_label.setGeometry(500, 20, 100, 100)
        
        self.msg_label.setGeometry(10, 65, 600, 50)
        
        
        layout_top.addWidget(btn_loadhand)
        layout_top.addWidget(btn_loadeffect)
        layout_top.addWidget(btn_recognition)
        layout_top.addWidget(btn_composeButton)
        layout_top.addWidget(btn_quit)
        layout_top.addWidget(x_slider)
        layout_top.addWidget(self.x_label)
        layout_top.addWidget(y_slider)
        layout_top.addWidget(self.y_label)
        layout_top.addWidget(self.msg_label)
        
        # Cam = QImage()  
        # layout_bottom.addWidget(Cam)
        
        #윈도우 창에 x, y위치와 w, h크기로 버튼와 레이블을 생성
        

        #각 버튼에 콜백함수 연결
        btn_loadhand.clicked.connect(self.loadhandfunction)
        btn_loadeffect.clicked.connect(self.loadeffectFunction)
        btn_recognition.clicked.connect(self.recognitionFunction)
        btn_composeButton.clicked.connect(self.composeFunction)
        btn_quit.clicked.connect(self.quitFunction)
        x_slider.valueChanged.connect(self.x_value_changed);
        y_slider.valueChanged.connect(self.y_value_changed);
        
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
    def x_value_changed(self,value):  
        self.x=value
        self.x_label.setText('위치이동 x '+str(self.x))
        # print(self.x)
        
    def y_value_changed(self,value):  
        self.y=value
        self.y_label.setText('위치이동 y '+str(self.y))
        # print(self.y)
        
    def loadhandfunction(self):
        self.msg_label.clear()
        self.msg_label.setText('이미지를 불러왔습니다.')

        global handImg, img_w, img_h ,x,y
         # 다이알로그로 파일의 경로를 가져옴
        handImg = QFileDialog.getOpenFileName(self,'','./')
        
        # 경로에 있는 이미지 읽어옴
        handImg = cv.flip(cv.imread(handImg[0]), 1)

        # 이미지 크기 재조정
        handImg = cv.resize(handImg, (0, 0), fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)
                
        # 이미지 크기 전역 변수에 저장
        img_w, img_h = handImg.shape[1], handImg.shape[0]
        
        # 저장된 배열 영상을 출력
        cv.imshow('myhnad', handImg)

    def loadeffectFunction(self):
        self.msg_label.clear()
        self.msg_label.setText('효과를 불러왔습니다')
        
        global effectImg, e_w, e_h, alpha
        # 다이알로그로 파일의 경로를 가져옴
        effectImg=QFileDialog.getOpenFileName(self,'','./')
        
        # 경로에 있는 이미지 읽어옴
        effectImg = cv.imread(effectImg[0],cv.IMREAD_UNCHANGED)

        # 이미지 크기 재조정
        effectImg=cv.resize(effectImg,(0,0),fx=0.2,fy=0.2,interpolation=cv.INTER_AREA)
        
        # 이미지 크기 전역 변수에 저장
        e_w, e_h = effectImg.shape[1], effectImg.shape[0]
        alpha=effectImg[:,:,3:]/255
    
        cv.imshow('Road Scene', effectImg)

    def recognitionFunction(self):    
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:
            
            # global handImg
            # global wirst
            
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = hands.process(cv.cvtColor(handImg, cv.COLOR_BGR2RGB))

            # 손으로 프린트하고 이미지에 손 랜드마크를 그립니다.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                print("손을 찾지 못함")
                exit(1)
            
            self.msg_label.clear()
            self.msg_label.setText('손이 인식되었습니다')
            
            global wrist
            wrist=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]  

            # mesh가 그려진 이미지를 저장
            # cv.imshow('Mesh Hand', cv.flip(annotated_image, 1))
            
    def composeFunction(self):

        global wrist
        annotated_image = handImg.copy()
            
        # 일부분 자르기 위한 좌표 얻기
        x1,x2=int(wrist.x*img_w-e_w/2)-self.x,int(wrist.x*img_w+e_w/2)-self.x
        y1,y2=int(wrist.y*img_h-e_h/2)-self.y,int(wrist.y*img_h+e_h/2)-self.y
    
        if x1>0 and y1>0 and x2<img_w and y2<img_h:
            annotated_image[y1:y2,x1:x2]=handImg[y1:y2,x1:x2]*(1-alpha)+effectImg[:,:,:3]*alpha
        
        # for hand_landmarks in results.multi_hand_landmarks:
          
           # 손가락 랜드마크 정보를 출력  
           # print('hand_landmarks:', hand_landmarks)
           # print(
           #      f'Index finger tip coordinates: (',
           #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img_w}, '
           #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img_h})'
           # )
           # mesh를 그리는 부분
           # mp_drawing.draw_landmarks(
           #    annotated_image,
           #    hand_landmarks,
           #    mp_hands.HAND_CONNECTIONS,
           #    mp_drawing_styles.get_default_hand_landmarks_style(),
           #    mp_drawing_styles.get_default_hand_connections_style())
           
        cv.imshow('Mesh Hand', cv.flip(annotated_image, 1))
            
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = InMyHand()
win.show()
app.exec_()


#%%
# while True:
# ret, frame = cap.read()
# if not ret:
# print('프레인 획득에 실패하여 루프를 나갑니다.')
# break

# res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
# if res.detections:
# for det in res.detections:
# p = mp_face_detection.get_key_point(
#     det, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
# x1, x2 = int(p.x*frame.shape[1]-w//2), int(p.x*frame.shape[1]+w//2)
# y1, y2 = int(p.y*frame.shape[0]-h//2), int(p.y*frame.shape[0]+h//2)
# alpha = dice[:, :, 3:]/255  # 투명도를 나타내는 알파값
# if x1 > 0 and y1 > 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
# frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]*(1-alpha)+dice[:, :, :3]*alpha
# cv.imshow('MediaPipe Face AR', cv.flip(frame, 1))
# if cv.waitKey(5) == ord('q'):
# break
