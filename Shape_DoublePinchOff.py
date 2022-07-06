# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:46:21 2021

@author: ZRJiang
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit
from numpy import linalg as LA
import copy
from scipy import integrate
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import itertools

# 將外部輪廓利用Canny存成excel
########################################參數區############################################


dPath: str = r'D:\Research\M7D01\M7D01_cylinder_xmm_peelingSpeed_0.1xcm.s_1'  # dir path
Filename_Extension = r'bmp' #filename extension

rrdis = 19  #real reference distance 參考物實際距離 #38 30.2 19 12.7 8

tStep = 1 #timeStep  

#######################################函數#############################################
# 把\(反斜線)換成/(斜線)


def convert(a):
    for i in range(len(a)):
        if a[i] == '\\':
            a = a[0:i]+r'/'+a[i+1:]
    return(a)

# 把檔案路徑讀給python


def loadImages(path=".", file_extension=r'bmp'):
    real_list = []
    real_list2 = []
    real_list3 = []
    list = os.listdir(path)
    for i in list:
        if(i.endswith(r'.' + file_extension)):
            for j in range(len(i)):
                if i[j] == r'.':
                    real_list.append(int(i[:j]))
                    real_list3.append((i[:j]))
            real_list2.append(path+'\\'+i)
    for i in range(len(real_list)):
        for j in range(len(real_list)):
            if real_list[i] < real_list[j]:
                temp = real_list[i]
                real_list[i] = real_list[j]
                real_list[j] = temp
                temp = real_list2[i]
                real_list2[i] = real_list2[j]
                real_list2[j] = temp
                temp = real_list3[i]
                real_list3[i] = real_list3[j]
                real_list3[j] = temp
    return real_list3, real_list2, real_list


def get_points(img, ratio=1, mode=1):
    global points, bList, temp, image, temptemp ,step
    step = 0

    def mouse_handler(event, x, y, flags, data):
        global x1, y1, step, temp, image, points, temptemp
        # mode1 choose reference points
        if mode == 1 and event == cv2.EVENT_LBUTTONDOWN:
            if step == 0:
                temp = copy.deepcopy(image)
            if step % 2 == 0:
                image = copy.deepcopy(temp)
                points[0] = (x, y)
            if step % 2 == 1:
                points[1] = (x, y)
            cv2.circle(image, (x, y), 3, (0, 0, 255), 5, 16)
            step = step + 1

        # mode2 select the cropping range
        if mode == 2 and event == cv2.EVENT_LBUTTONDOWN:
            image = copy.deepcopy(temp)
            temptemp = copy.deepcopy(image)
            x1, y1 = x, y
            points[2] = (x, y)
        elif mode == 2 and event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            image = copy.deepcopy(temptemp)
            cv2.rectangle(image, (x1, y1), (x, y), (0, 0, 255), 1)
        elif mode == 2 and event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(image, (x1, y1), (x, y), (0, 0, 255), 1)
            points[3] = (x, y)

        # mode3
        if mode == 3 and event == cv2.EVENT_LBUTTONDOWN:
            #print(point[4] , point[5])
            image = copy.deepcopy(temp)
            print(step)
            if step % 2 == 0:
                points[4] = (x, y)
            if step % 2 == 1:
                points[5] = (x, y)
            if points[4] != []:
                cv2.drawMarker(image, points[4], (255, 255, 255))
            if points[5] != []:
                cv2.drawMarker(image, points[5], (255, 255, 255))
            step = step + 1

        # mode4
        if mode == 4 and event == cv2.EVENT_LBUTTONDOWN:
            temptemp = copy.deepcopy(image)
            x1, y1 = x, y
            bList.append((x+points[2][0], y+points[2][1]))
        elif mode == 4 and event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            image = copy.deepcopy(temptemp)
            cv2.rectangle(image, (x1, y1), (x, y), (0, 0, 255), 1)
        elif mode == 4 and event == cv2.EVENT_LBUTTONUP:
            if x < 0:
                x = 0
            if x > image.shape[1]:
                x = image.shape[1]
            if y < 0:
                y = 0
            if y > image.shape[0]:
                y = image.shape[0]
            cv2.rectangle(image, (x1, y1), (x, y), (255, 255, 255), -1)
            bList.append((x+points[2][0], y+points[2][1]))

    # 建立一個 window
    image = img
    # temp=copy.deepcopy(image)
    cv2.namedWindow("Image", 0)
    # 改變 window 成為適當圖片大小
    h, w = img.shape[0], img.shape[1]

    if (mode == 2 or mode == 3) and points[0] != []:
        cv2.circle(img, points[0], 3, (0, 0, 255), 5, 16)
        cv2.circle(img, points[1], 3, (0, 0, 255), 5, 16)
        cv2.rectangle(img, points[2], points[3], (0, 0, 255), 1, 16)
        temp = copy.deepcopy(image)
        # image=copy.deepcopy(temp)

    #print("Img height, width: ({}, {})".format(h, w))
    cv2.resizeWindow("Image", ratio*w, ratio*h)

    # 利用滑鼠回傳值，資料皆保存於 data dict中
    cv2.setMouseCallback("Image", mouse_handler)

    # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
    while(1):

        cv2.imshow('Image', image)
        #cv2.moveWindow("Image", 100, 100)
        k = cv2.waitKey(1)

        if k == ord('1'):
            print("mode1 choose reference points")
            mode = 1
            step = 0

        elif k == ord('2'):
            print("mode2 select the cropping range")
            mode = 2
            temp = copy.deepcopy(image)

        elif k == ord('3'):
            print("mode3")
            mode = 3
            step = 0
            temp = copy.deepcopy(image)

        elif k == ord('4'):
            mode = 4
            print("mode4 ")

        elif k == ord('q'):
            if points[0] != []  and points[4] != []:
                break

        elif k == ord('r'):
            if mode == 4:
                image = copy.deepcopy(temptemp)
                bList.pop()
                bList.pop()
        elif k == 27:
            cv2.destroyAllWindows()
            return -1

    cv2.destroyAllWindows()
    # 回傳點 list


################################## 主程式 ################################################

# filenameList , filepathList , intfilenameList
fList , pList , ifList = loadImages(dPath,Filename_Extension) 
bList = [] #blockList  用來儲存白色的區域
points = [ [], [], [], [], [], [], [] ]

#判斷資料夾是否存在
if (os.path.exists(convert(dPath)+'\/edge') == 0):
    os.mkdir(convert(dPath)+'\/edge')
    
    
###將圖片修圖並且存檔
for i in range(len(fList)):
    origin = cv2.imread(pList[i] )
    canny = cv2.Canny(origin, 150, 200)
    
    if points[0] == []:
        control = 1
    else:
        control = 3
    if get_points(origin, 1, control) == -1:
        break
    
    if points[2] == [] or points[3] == []:
        points[2] = canny.shape[1]
        points[3] = canny.shape[0]

    img = canny[points[2][1]:points[3][1], points[2][0]:points[3][0]]
    img = 255 - img
    
    for j in range(len(bList)//2):
        cv2.rectangle(img, (bList[2*j][0]-points[2][0], bList[2*j][1]-points[2][1])\
                      ,(bList[2*j+1][0]-points[2][0], bList[2*j+1][1]-points[2][1]), 255, -1)
    if get_points(img, 2, 4) == -1:
        break

    print("complete %d of %d" % (i+1, len(fList)))
    cv2.imwrite(dPath+'\\edge\\' +
                fList[i]+r'.'+Filename_Extension, img)




fList , pList , ifList = loadImages(dPath+r'\edge',Filename_Extension)
rminList = []
excel = pd.ExcelWriter(dPath+r'\data.xlsx')
for i in range(len(fList)):
    lEdge , rEdge , r  = [] , [] ,[]
    data = {}
    img = cv2.imread(pList[i] , cv2.IMREAD_GRAYSCALE)
    for y in range( img.shape[0] ):
        tList, = np.where(img[y] == 0)
        lEdge.append( np.min(tList) )
        rEdge.append( np.max(tList) )
        r.append( (np.max(tList)-np.min(tList))//2 )
        
    # 找到中心位置
    mean = np.mean(r)
    std = np.std(r)
    tList = [] 
    for j in range(len(r)):
        if r[j]<mean+3*std and r[j]>mean-3*std:
            tList.append(r[j])
    rmin = min(tList)
    tList, = np.where(r == rmin)
    x0 = int(np.average(tList))
    x = np.linspace(1, len(r), len(r))
    x = x-x0
    
    
    lEdge = np.array(lEdge)
    rEdge = np.array(rEdge)
    r = np.array(r)
    data = {'x':x , 'r':r , 'left':lEdge , 'right':rEdge}
    data = pd.DataFrame(dict([(k1, pd.Series(v)) for k1, v in data.items()]))
    temp = pd.DataFrame({'r': [points[5][1]-points[4][1], rrdis],
                         'left': [points[4][1]-points[2][1]-x0, points[0][0]],
                         'right': [points[5][1]-points[2][1]-x0, points[1][0]]}, index=['height', 'reference'])
    data = pd.concat([data, temp])
    data.to_excel(excel, sheet_name=fList[i])
    rminList.append(r[x0])
excel.save()
del excel

fminList = copy.deepcopy(rminList) #final rminList
fminList = np.array(fminList)
fminList = fminList * rrdis/(points[1][0]-points[0][0])
ifList = np.array(ifList)
for i in range(len(ifList)):
    ifList[i] = (ifList[-1]-ifList[i])
plt.title('x to r_min')
plt.xlabel('x')
plt.ylabel('r_min')
plt.plot(ifList, fminList)
plt.show()
excel = pd.ExcelWriter(dPath+'\\'+'Rmin.xlsx')
data = {'frames': fList, 'r_min(pixcel)': rminList, 't(ms)': ifList/1000, '='+str(
    rrdis)+'/('+str(points[1][0])+'-'+str(points[0][0])+')': fminList}
data = pd.DataFrame(data=data)
data.to_excel(excel)
excel.save()
del excel