#!/usr/bin/env python

import numpy as np
import cv2
import os
import sqlite3




# Set the right value for the directory name, and create the directory
dir_id = 1
while os.path.isdir(os.path.join('bdd_faces', 's' + f'{dir_id}')) == True:
    dir_id += 1
os.makedirs(os.path.join('bdd_faces', 's' + f'{dir_id}'))

# Take 10 .pgm pictures and save them in the new directory
for i in range(1, 11): 
    
    cap = cv2.VideoCapture(0)

    # Capture a frame using spacebar
    while(cap.isOpened()):
        ret, picture = cap.read()
        ret, frame = cap.read()
        mirror = cv2.flip(frame, 1)
        cv2.rectangle(mirror, (134, 14), (506, 466), (0, 0, 255), 2)
        cv2.circle(mirror, (247, 227), 20, (0, 0, 255), 1)
        cv2.circle(mirror, (392, 227), 20, (0, 0, 255), 1)
    
        if ret==True:
            cv2.imshow(f'Appuez sur Espace - Press Space - Photo {i}/10', mirror)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                image = picture
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    
    # Crop picture and convert to greyscale
    cropped = image[16:464, 136:504]
    grey = cv2.cvtColor( cropped, cv2.COLOR_RGB2GRAY )
    
    
    # Downsize and save picture
    dim = (92, 112)
    resized = cv2.resize(grey, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join('bdd_faces', 's' + f'{dir_id}',  f'{i}.pgm'), resized, [cv2.IMWRITE_PXM_BINARY, 0])

# Open names.db to write the name of the new entry
os.system("names.db")


