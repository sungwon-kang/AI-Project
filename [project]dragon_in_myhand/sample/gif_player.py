import cv2

# gif 크기 재조정
# https://ezgif.com/resize/ezgif-5-17a06701e3.gif

gifPath="./gif/fire.gif"

gif = cv2.VideoCapture(gifPath)
# ret, frame = gif.read()  ret=True if it finds a frame else False.
#%%
while cv2.waitKey(33)<0:
    
    if gif.get(cv2.CAP_PROP_POS_FRAMES)==gif.get(cv2.CAP_PROP_FRAME_COUNT):
        gif.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, frame = gif.read()
    cv2.imshow("VideoFrame",frame)

gif.release()
cv2.destoryAllWindows()
#%%
