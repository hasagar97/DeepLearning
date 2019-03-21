import numpy as np
import cv2
import os
from PIL import Image
import sys

frames = []

from os.path import isfile, join

def convert_frames_to_video(pathOut,fps,frame_array):

    
    height, width, layers = frame_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    

def translate(l, w, a, c):    # (length, width, angle, color)

    length = 7
    if l == 1:
        length = 15

    width = 1
    if w == 1:
        width = 3

    angle = (15*a)*np.pi/180

    color = 0
    if c == 1:
        color = 2

    loc = str(l) + '_' + str(w) + '_' + str(a) + '_' + str(c)

    imgArray = []
    count = 0
    imgCount = 0

    while count < 1000:
        for i in range(28):
            for j in range(28):
                if count >= 1000:
                    break
                check = 1
                img = np.zeros([28,28,3],dtype=np.uint8)
                for k in range(0,length+1):
                    x =  (i + int(round(k*np.cos(angle))))
                    y =  (j - int(round(k*np.sin(angle))))
                    
                    if x < 0 or x >= 28 or y < 0 or y >= 28:
                        check = 0
                        break
                    
                    if width == 1:
                        img[y][x][color] = 255

                    else:
                        if angle == np.pi/2 and x > 1 and x < 26:
                            img[y][x-1][color] = img[y][x][color] = img[y][x+1][color] = 255

                        elif y > 1 and y < 26:
                            img[y-1][x][color] = img[y-1][x][color] = img[y-1][x][color] = 255
                        
                        else:
                            check = 0
                            break

                if check == 1:
                    pic = Image.fromarray(img)
                    count += 1
                    pic.save('Images/' + loc + '_' + str(count) + '.jpg',  quality = 100000)
                    if count % 11 == 0:
                        imgArray.append(img)
                        imgCount += 1
                    if imgCount == 9:
                        imgCount = 0
                        frames.append(imgArray)
                        imgArray = []


try:  
    os.mkdir("./Images")
    
except:
    temp123 = 1

for length in range(2):
    for width in range(2):
        for angle in range(12):
            for color in range(2):
                translate(length, width, angle, color)





frames_array=[]
for array in frames:
    img = np.zeros([28*3,28*3,3],dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            img[i*28:i*28+28,j*28:j*28+28]=array[i+3*j] 
    frames_array.append(img)

pathOut = 'video.avi'
fps = 2.0
convert_frames_to_video(pathOut, fps,frames_array)