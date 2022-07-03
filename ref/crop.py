from PIL import Image
 
image1 = Image.open('./3_CaptureSample/test.png')
# image1.show()
 
#이미지의 크기 출력
width, height = image1.size 

init_n=2    # 초기 이미지 조정
n=5         # 이미지 분할 수
a=8       # x1 조절
b=12         # x2 조절

print(width, height)

width=width/init_n
print(width)
# 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
for i in range (n):
    w = width/n*i
    croppedImage=image1.crop((w+a ,0, width/n*(i+1) + b ,height))
    print("잘려진 사진 크기 :",croppedImage.size)
    # croppedImage.show()
    croppedImage.save('./3_CaptureSample/croppedSample/test'+str(i)+'.png')
    
 