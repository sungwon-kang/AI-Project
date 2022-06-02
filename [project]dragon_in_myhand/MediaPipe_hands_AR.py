#%%
# 메인 기능
# 1. 손 사진에 메쉬 얻기  ok
# 2. 특정 위치에 원하는 이미지 합성하기  ok


# 부가 기능 - 원하는 사진을 불러올 수 있게 하기
#          - 원하는 위치, 크기를 효과 이미지를 조정할 수 있게 하기

# 완성 기능
# 1. 캠을 화면에 출력하기
# 2. 캠에 효과 이미지가 손을 따라가기
# 3. 이미지를 GIF나 애니메이션으로 만들기
#%%
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

width = 1024
height = 768
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
    effect=cv.resize(effect,(0,0),fx=0.2,fy=0.2,interpolation=cv.INTER_AREA)
    e_w, e_h = effect.shape[1], effect.shape[0]
    img_w, img_h = image.shape[1], image.shape[0]

    annotated_image = image.copy()
    #%% 손등 위치 가져오기
    # 약지 시작부분 위치
    # FINGER=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # 손목
    p=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
    
    # 일부분 자르기 위한 좌표 얻기
    x1,x2=int(p.x*image.shape[1]-e_w/2),int(p.x*image.shape[1]+e_w/2)
    y1,y2=int(p.y*image.shape[0]-e_h/2),int(p.y*image.shape[0]+e_h/2)
    
    # 효과이미지 알파값 정규화해서 저장
    alpha=effect[:,:,3:]/255
    
    # 손 이미지와 효과 이미지 합성
    if x1>0 and y1>0 and x2<image.shape[1] and y2<image.shape[0]:
        annotated_image[y1:y2,x1:x2]=image[y1:y2,x1:x2]*(1-alpha)+effect[:,:,:3]*alpha
       
    
    # 원본 이미지에서 얻은 손가락 랜드마크에 따라 mesh를 그림
    for hand_landmarks in results.multi_hand_landmarks:
      
       # 손가락 랜드마크 정보를 출력  
       # print('hand_landmarks:', hand_landmarks)
       # print(
       #      f'Index finger tip coordinates: (',
       #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img_w}, '
       #      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img_h})'
       # )
       # mesh를 그리는 부분
       mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    
    # mesh가 그려진 이미지를 저장
    cv.imwrite(
        'mesh_myhand.png', cv.flip(annotated_image, 1))

