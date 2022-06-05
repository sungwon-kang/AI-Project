import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import winsound

#%%
# UI 구성 ok
# 다이알로그 이미지 불러오기 ok
# 불러온 이미지 하단 프레임에 출력하기
# 손 인식하기 기능 넣기 + 인식이 안될 시 예외처리
# 이미지 합성 기능 넣기
# 사용법 버튼 만들기 + 내용 채우기
#%%


class InMyHand(QMainWindow):

    def __init__(self):
        # 메인 윈도우 창 크기와 타이틀 지정
        super().__init__()
        self.setWindowTitle('내 손안에 흑염룡이')
        self.setGeometry(200, 200, 800, 500)
        
        main_layout=QVBoxLayout()
        
        frame_top = QFrame(self)
        frame_top.setGeometry(0,0,800,75)
        frame_top.setFrameShape(QFrame.Box | QFrame.Plain)
        
        frame_bottom = QFrame(self)
        frame_bottom.setGeometry(0,75,800,435)
        frame_bottom.setFrameShape(QFrame.Box | QFrame.Plain)
        
        layout_top = QVBoxLayout()
        layout_bottom = QVBoxLayout()
        
        # 버튼와 레이블 선언
        btn_loadhand = QPushButton('손 불러오기', self)
        btn_loadeffect = QPushButton('효과 불러오기', self)
        btn_recognition = QPushButton('손 인식하기', self)
        btn_composeButton = QPushButton('이미지 합성', self)
        btn_quit = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)
        
        layout_top.addWidget(btn_loadhand)
        layout_top.addWidget(btn_loadeffect)
        layout_top.addWidget(btn_recognition)
        layout_top.addWidget(btn_composeButton)
        layout_top.addWidget(btn_quit)
        
        Cam = QImage()  
        layout_bottom.addWidget(Cam)
        
        #윈도우 창에 x, y위치와 w, h크기로 버튼와 레이블을 생성
        btn_loadhand.setGeometry(10, 10, 100, 30)
        btn_loadeffect.setGeometry(110, 10, 100, 30)
        btn_recognition.setGeometry(210, 10, 100, 30)
        btn_composeButton.setGeometry(310, 10, 100, 30)
        btn_quit.setGeometry(700, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 50)

        #각 버튼에 콜백함수 연결
        btn_loadhand.clicked.connect(self.loadhandfunction)
        btn_loadeffect.clicked.connect(self.loadeffectFunction)
        btn_recognition.clicked.connect(self.recognitionFunction)
        btn_composeButton.clicked.connect(self.composeFunction)
        btn_quit.clicked.connect(self.quitFunction)

        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        
    def loadhandfunction(self):
        self.label.clear()
        self.label.setText('이미지를 불러왔습니다.')

        # singFiles에 있는 png 파일들을 signImgs 리스트에 배열 영상으로 저장
        handimg = QFileDialog.getOpenFileName(self,'','./')
        
        handimg = cv.imread(handimg[0])

        handimg = cv.resize(handimg, (0, 0), fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)

        # 저장된 배열 영상을 출력
        cv.imshow('myhnad', handimg)

    def loadeffectFunction(self):
        self.label.clear()
        self.label.setText('효과를 불러왔습니다')
        
        effectImg=QFileDialog.getOpenFileName(self,'','./')
        # print(fname[0])
        # 다이알로그가 잘 작동하지 않아 영상을 지정
        self.effectImg = cv.imread(effectImg[0])

        # if self.roadImg is None:
        #     sys.exit('파일을 찾을 수 없습니다.')

        # 거리 영상을 출력
        cv.imshow('Road Scene', self.effectImg)

    def recognitionFunction(self):
        print()
        
    def composeFunction(self):
        print()
            
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