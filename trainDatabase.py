#!/usr/bin/env python

import os
import cv2
import sys
import shutil
import random
import operator
import sqlite3
import pickle
import numpy as np

"""
Train and save the eigenvectors for a specified database.

Example Call:
    $> python eigenfaces.py bdd_faces
"""
class Eigenfaces(object):                                                       # *** COMMENTS ***
    faces_count = 0
    faces_dir = '.'                                                             # directory path to the faces database

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

        self.faces_dir = _faces_dir
        self.faces_count = len([f for f in os.listdir(self.faces_dir)]) - 1
        self.energy = _energy
        self.training_ids = []                                                  # train image id's for every at&t face
        self.l = self.train_faces_count * self.faces_count                                         # training images count
        
        L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image

        cur_img = 0
        for face_id in range(1, self.faces_count + 1):

            training_ids = random.sample(range(1, 11), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                              # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(training_id) + '.pgm')          # relative path

                img = cv2.imread(path_to_img, 0)                                # read a grayscale image
                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d

                L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
                cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l                          # get the mean of all images / over the rows of L

        for j in range(0, self.l):                                             # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as
        C /= self.l                                                             # L*L^T, we set C = L^T*L, and end up with way
                                                                                # smaller and computentionally inexpensive one
                                                                                # we also need to divide by the number of training
                                                                                # images


        self.evalues, self.evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
        sort_indices = self.evalues.argsort()[::-1]                             # getting their correct order - decreasing
        self.evalues = self.evalues[sort_indices]                               # puttin the evalues in that order
        self.evectors = self.evectors[sort_indices]                             # same for the evectors

        evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
        evalues_count = 0                                                       # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[0:evalues_count]

        self.evectors = self.evectors.transpose()                               # change eigenvectors from rows to columns
        self.evectors = L * self.evectors                                       # left multiply to get the correct evectors
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors

        self.W = self.evectors.transpose() * L                                  # computing the weights


    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        return (closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)

    """
    Evaluate the model using the 4 test faces left
    from every different face in the database.
    """
    def evaluate(self):
        print ('> Auto-evaluation of database')
        results_file = os.path.join('results', 'att_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = self.test_faces_count * self.faces_count                   # number of all AT&T test images/faces
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (int(result_id) == int(face_id))

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        self.accuracy = float(100. * test_correct / test_count)
        print ('---> Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

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
            top5_ids = np.argpartition(norms, 5)[:5]                           # first five elements: indices of top 5 matches in AT&T set

            name_noext = os.path.splitext(img_name)[0]                          # the image file name without extension
            result_dir = os.path.join('results', name_noext)                    # path to the respective results folder
            os.makedirs(result_dir)                                             # make a results folder for the respective celebrity
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
            #+ '<att faces dir> [<celebrity faces dir>]'
        sys.exit(1)

    if not os.path.exists('results'):                                           # create a folder where to store the results
        os.makedirs('results')
    else:
        shutil.rmtree('results')                                                # clear everything in the results folder
        os.makedirs('results')

    efaces = Eigenfaces(str(sys.argv[1]))                                       # create the Eigenfaces object with the data dir
    efaces.evaluate()                                                           # evaluate our model

    # Saving the object containing the eigenvectors:
    with open('database.pkl', 'wb') as f:
        pickle.dump(efaces, f, pickle.HIGHEST_PROTOCOL)
    print('> Eigenvalues saved in database.pkl')

    #if len(sys.argv) == 3:                                                      # if we have third argument (celebrity folder)
    #    efaces.evaluate_faces(str(sys.argv[2]))                           # find best matches for the celebrities

