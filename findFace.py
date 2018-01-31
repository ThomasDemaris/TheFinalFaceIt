#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import os
import cv2
import sys
import shutil
import random
import operator
import sqlite3
import pickle
import datetime
import numpy as np

"""
A Python class that implements the Eigenfaces algorithm
for face recognition, using eigenvalue decomposition and
principle component analysis.

We use the AT&T data set, with 60% of the images as train
and the rest 40% as a test set, including 85% of the energy.

Additionally, we use a small set of celebrity images to
find the best AT&T matches to them. 

Example Call:
    $> python eigenfaces.py bdd_faces test_faces

Algorithm Reference:
    http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""
class Eigenfaces(object):                                                       # *** COMMENTS ***
    faces_count = 0
    faces_dir = '.'                                                             # directory path to the AT&T faces

    train_faces_count = 9                                                       # number of faces used for training
    test_faces_count = 1                                                        # number of faces used for testing

    l = 0
    m = 92                                                                      # number of columns of the image
    n = 112                                                                     # number of rows of the image
    mn = m * n                                                                  # length of the column vector

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = '.', _energy = 0.85):
       #Read eigenfaces database
        with open('database.pkl', 'rb') as input:
            efaces = pickle.load(input)
        self.mean_img_col = efaces.mean_img_col
        self.mn = efaces.mn
        self.evectors = efaces.evectors
        self.W = efaces.W
        self.faces_dir = efaces.faces_dir
        self.faces_count = efaces.faces_count
        self.training_ids = efaces.training_ids

    """
    Evaluate the model for the small celebrity data set.
    Returning the top 5 matches within the database.
    Images should have the same size (92,112) and are
    located in the test_dir folder.
    """
    def evaluate_faces(self, test_dir='.'):
        print ('> Evaluating test faces')
        for img_name in os.listdir(test_dir):                              # go through all the celebrity images in the folder
            path_to_img = os.path.join(test_dir, img_name)

            img = cv2.imread(path_to_img, 0)                                    # read as a grayscale image
            img_col = np.array(img, dtype='float64').flatten()                  # flatten the image
            img_col -= self.mean_img_col                                        # subract the mean column
            img_col = np.reshape(img_col, (self.mn, 1))                         # from row vector to col vector

            S = self.evectors.transpose() * img_col                             # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

            diff = self.W - S                                                   # finding the min ||W_j - S||
            norms = np.linalg.norm(diff, axis=0)
            top5_ids = np.argpartition(norms, 5)[:5]                           # first five elements: indices of top 5 matches in database

            name_noext = os.path.splitext(img_name)[0]                          # the image file name without extension

            #Create corresponding result directory
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d_%Hh%Mm%Ss_")
            result_dir = '.\\results\\'+date+img_name
            os.makedirs(result_dir)
            
            #Create text file to store results
            result_file = os.path.join(result_dir, 'results.txt')               # the file with the similarity value and id's

            f = open(result_file, 'w')                                          # open the results file for writing

            topid_tuples = []

            for top_id in top5_ids:
                face_id = int(top_id / self.train_faces_count) + 1                 # getting the face_id of one of the closest matches
                subface_id = self.training_ids[face_id-1][top_id % self.train_faces_count]           # getting the exact subimage from the face

                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(subface_id) + '.pgm')           # relative path to the top5 face

                shutil.copyfile(path_to_img,                                    # copy the top face from source
                        os.path.join(result_dir, str(top_id) + '.pgm'))         # to destination
                
                topid_tuples.append(tuple((top_id, norms[top_id])))
            topid_tuples.sort(key=operator.itemgetter(1)) #sort by score

            best_face_id = 0
            best_subface_id = 0

            for i in range(0, 5):
                global_id = topid_tuples[i][0]
                score = topid_tuples[i][1]
                found_face_id = int(global_id / self.train_faces_count + 1)
                if (i==0):
                    best_face_id = found_face_id
                    best_subface_id = self.training_ids[found_face_id-1][global_id % self.train_faces_count]

                 #Find name in database
                conn = sqlite3.connect('names.db')
                c = conn.cursor()
                c.execute('SELECT Name FROM Names WHERE Id='+str(found_face_id))
                user1 = c.fetchone()
                f.write('%3d. found face id: %3d, name: %s, pic id: %3d, score: %.6f\n' % (i+1, found_face_id, user1[0], global_id, score))     # write the id and its score to the results file

            f.close()                                                           # close the results file
        
            #Show best match for last picture taken
            last_picture_id=1
            while os.path.isfile(os.path.join('test_faces', f'{last_picture_id}.jpg')) == True:
                last_picture_id += 1
            last_picture_id -= 1
                
            path_to_best = os.path.join(self.faces_dir, 's'+str(best_face_id), str(best_subface_id)+'.pgm')
            if (os.path.join('test_faces', f'{img_name}')) == (os.path.join('test_faces', f'{last_picture_id}.jpg')):
                best_match = cv2.imread(path_to_best)
                window_name = "Best match for " + str(img_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, best_match)
                cv2.resizeWindow(window_name, 184,224)
                cv2.waitKey(0)  
                cv2.destroyAllWindows()
            
        print ('---> Evaluation done: check results directory')


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        #print 'Usage: python2.7 eigenfaces.py ' \
            #+ '<att faces dir> [<test faces dir>]'
        sys.exit(1)

    if not os.path.exists('results'):                                           # create a folder where to store the results
        os.makedirs('results')
    else:
        shutil.rmtree('results')
        #os.makedirs('results')                                                # clear everything in the results folder

    eigenfaces = Eigenfaces();
    eigenfaces.evaluate_faces(str(sys.argv[1]))                           # find best matches for the celebrities

