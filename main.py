import os
# from PIL import Image
import csv
import cv2
import sys
import time

#sober参数
dx = 0
dy = 0
# *****canny算子参数××××
max = 149
min = 49

# *****Laplacian算子参数××××
ksize1 = 1

csv_reader = csv.reader(open("./path.csv"))
for row in csv_reader:
    if row.__str__()[2:10] == 'readpath':
        data_dir = row.__str__()[11:-2]
    elif row.__str__()[2:10] == 'savepath':
        savepath = row.__str__()[11:-2]
    elif row.__str__()[2:4] == 'dx':
        dx = int(row.__str__()[5:-2])
    elif row.__str__()[2:4] == 'dy':
        dy = int(row.__str__()[5:-2])
    elif row.__str__()[2:5] == 'max':
        max = int(row.__str__()[6:-2])
    elif row.__str__()[2:5] == 'min':
        min = int(row.__str__()[6:-2])
    elif row.__str__()[2:7] == 'ksize':
        ksize1 = int(row.__str__()[8:-2])
    else:
        continue

classes = os.listdir(data_dir)
i=0
total = classes.__len__()
for cls in classes:
    img_src = cv2.imread(data_dir+cls, 1)
    gray_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    if not os.path.exists(savepath+"gray/"):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
        os.makedirs(savepath+"gray/")
    cv2.imwrite(savepath+"gray/"+cls, gray_img)

    #sobel边缘检测
    edges = cv2.Sobel(img_src, cv2.CV_16S, dx, dy)
    if not os.path.exists(savepath +"sobel/"):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
        os.makedirs(savepath +"sobel/")
    cv2.imwrite(savepath +"sobel/"+ cls, edges)

    #canny边缘检测
    canny = cv2.Canny(gray_img, min, max)     # 调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
    if not os.path.exists(savepath + "canny/"):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
        os.makedirs(savepath + "canny/")
    cv2.imwrite(savepath + "canny/" + cls, canny)

    #lapacian边缘检测
    dst_img = cv2.Laplacian(img_src, cv2.CV_32F, ksize=ksize1)
    laplacian_edge = cv2.convertScaleAbs(dst_img)  # 取绝对值后，进行归一化
    if not os.path.exists(savepath + "laplacian/"):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
        os.makedirs(savepath + "laplacian/")
    cv2.imwrite(savepath + "laplacian/" + cls, laplacian_edge)
    i = i+1
    sys.stdout.write('\r%s%%' % (i/total*100))
    sys.stdout.flush()
sys.stdout.write("\n")
sys.stdout.write("finish!")



