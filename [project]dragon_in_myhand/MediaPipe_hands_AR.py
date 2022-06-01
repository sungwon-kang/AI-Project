import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# fire=cv.imread('./gif/fire.gif',cv.IMREAD_UNCHANGED) # 증강 현실에 쓸 장신구
# fire=cv.resize(fire,dsize=(0,0),fx=0.1,fy=0.1)
# w,h=fire.shape[1],fire.shape[0]

# 이미지 파일의 경우을 사용하세요.:
IMAGE_FILE = './img/myhand.png'
IMAGE_EFFECT ='./img/fire.png'

#%%
#코드 출처 https://puleugo.tistory.com/10

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    # 이미지를 읽어 들이고, 보기 편하게 이미지를 좌우 반전합니다.
    image = cv.flip(cv.imread(IMAGE_FILE), 1)
    effect = cv.imread(IMAGE_EFFECT,cv.IMREAD_UNCHANGED)
    
    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
    results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # 손으로 프린트하고 이미지에 손 랜드마크를 그립니다.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        print("손을 찾지 못함")
        exit(1)
        
    # 이미지 크기를 재조정하고 저장합니다.
    
    image=cv.resize(image,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_AREA)
    effect=cv.resize(effect,(0,0),fx=0.5,fy=0.5,interpolation=cv.INTER_AREA)
    
    alpha=effect[:,:,3:]/255
    print('image=',image.shape)
    print('effect=', effect.shape)
    # 원본 이미지를 복사합니다.
    annotated_image = image.copy()
    # print(alpha)
    #%% 손등 위치 가져오기
    p=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
    FINGER=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # x1=WRIST.x * image_width
    # y1=WRIST.y * image_height
    
    # x2=FINGER.x * image_width
    # y2=FINGER.y * image_height
    
    x1,x2=int(p.x*image.shape[1]-effect.shape[1]//2),int(p.x*image.shape[1]+effect.shape[1]//2)
    y1,y2=int(p.y*image.shape[0]-effect.shape[0]//2),int(p.y*image.shape[0]+effect.shape[0]//2)
    print(x1,x2)
    print(y1,y2)
    
    # print(image[y1:y2,x1:x2].shape)
    # print(image[y1:y2,x1:x2].shape*(1-alpha)+effect[:,:,:3]*alpha)
    if x1>0 and y1>0 and x2<image.shape[1] and y2<image.shape[0]:
        image[y1:y2,x1:x2]=image[y1:y2,x1:x2]*(1-alpha)+effect[:,:,:3]*alpha
    
    
    cv.imshow('MediaPipe Face AR',image)
    cv.waitKey()
    cv.destoryAllWindows()
       
    #%%
    
    # 원본 이미지에서 얻은 손가락 랜드마크에 따라 mesh를 그립니다.
    for hand_landmarks in results.multi_hand_landmarks:
      
      # 손가락 랜드마크 정보를 출력  
      # print('hand_landmarks:', hand_landmarks)
      # print(
      #      f'Index finger tip coordinates: (',
      #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
      #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      # )
      # mesh를 그리는 부분
       mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    
    # mesh가 그려진 이미지를 저장합니다.
    cv.imwrite(
        'mesh_myhand.png', cv.flip(annotated_image, 1))

# # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
# cap = cv.VideoCapture(0)
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("카메라를 찾을 수 없습니다.")
#       # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
#       continue

#     # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
#     image.flags.writeable = False
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # 이미지에 손 주석을 그립니다.
#     image.flags.writeable = True
#     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     #보기 편하게 이미지를 좌우 반전합니다.
#     cv.imshow('MediaPipe Hands', cv.flip(image, 1))
#     if cv.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
