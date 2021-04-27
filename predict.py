'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import os
from PIL import Image

from yolo import YOLO
import cv2

yolo = YOLO()

img_dir=(r"E:/document/syn/data/graduation-project/yolo3-pytorch-master/img/01")
img_temp_dir = os.path.join(img_dir)
            #获取该目录下所有的文件
img_list = os.listdir(img_temp_dir)
for img_name in img_list:
    try:
        if not os.path.isdir(img_name):
                # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            img_path = os.path.join(img_temp_dir,img_name)
            image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()        

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()
        ##单个检测,需要保存时使用后续代码##
        # k= input("save?: ")
        # if k==str('y'): # 按下y键时保存并退出
        #     r_image.save("C:/Users/bamae/Desktop/1.jpg")
        #     print('save successfully')
