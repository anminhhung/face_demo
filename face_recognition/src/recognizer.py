from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time
from scipy import misc

def recognition(IMG_PATH, model, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet):
    #Time start
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.7
    INPUT_IMAGE_SIZE = 160
    name = "Unknow"
    prob = np.zeros(2)
    image = misc.imread(IMG_PATH)
    bounding_boxes, _ = align.detect_face.detect_face(image, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    faces_found = bounding_boxes.shape[0]
    
    if faces_found > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((faces_found, 4), dtype=np.int32)
        for i in range(faces_found):
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            # crop face detected
            cropped = image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
            if np.all(np.array(cropped.shape)):
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            
                # classifier
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            
                # if best_class_probabilities > 0.5:
                #     name = class_names[best_class_indices[0]]
                # else:
                #     name = "Unknown"
                name = class_names[best_class_indices[0]]
                prob = best_class_probabilities
    return name, prob

