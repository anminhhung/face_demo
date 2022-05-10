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


CLASSIFIER_PATH = 'Models/facemodels.pkl'

FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

# Load train model (classifier)
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)


with tf.Graph().as_default():

    # Set up GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        # Load model MTCNN
        facenet.load_model(FACENET_MODEL_PATH)

        # Get tensor input and output
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        # Set up NN
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

print("p-net: ", pnet)
print("r-net: ", rnet)
print("o-net: ", onet)
print("images-placeholder: ", images_placeholder)
print("phase-train-placeholder: ", phase_train_placeholder)
print("embeddings: ", embeddings)
print("sess: ", sess)
print("class-names: ", class_names)