#!/usr/bin/env python

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

#Set the right value for the filename
n=1
while os.path.isfile(os.path.join('test_faces', f'{n}.jpg')) == True:
                n += 1

#Capture a frame using spacebar
while(cap.isOpened()):
    ret, picture = cap.read()
    ret, frame = cap.read()
    mirror = cv2.flip(frame, 1)
    cv2.rectangle(mirror, (134, 14), (506, 466), (0, 0, 255), 2)
    cv2.circle(mirror, (247, 227), 20, (0, 0, 255), 1)
    cv2.circle(mirror, (392, 227), 20, (0, 0, 255), 1)
    
    
    if ret==True:
        cv2.imshow('Appuyez sur Espace - Press Space', mirror)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image = picture
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

#Crop picture
cropped = image[16:464, 136:504]

#Downsize and save picture
dim = (92, 112)
resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite(os.path.join('test_faces', f'{n}.jpg'), resized)


#Launch test script
os.system('python eigenfaces.py bdd_faces test_faces')
