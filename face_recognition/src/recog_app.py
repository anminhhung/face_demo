from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import tensorflow as tf
import argparse
import os
import sys
import math
import pickle
import detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64
import facenet
import time
import recognizer
import mp4toimage
import align_dataset_mtcnn_update
import classifier_update

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.7
# IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'DataSet/FaceData/Model/facemodels.pkl'
FACENET_MODEL_PATH = 'DataSet/FaceData/Model/20180402-114759.pb'

# with open(CLASSIFIER_PATH, 'rb') as f:
#     model, class_names = pickle.load(f)
# print("Custom Classifier, Successfully loaded")

tf.Graph().as_default()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options, log_device_placement=False))


# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = detect_face.create_mtcnn(sess, "src/align")

app = Flask(__name__)
#CORS(app)

@app.route('/')
@cross_origin()
def index():
    return "OK!"


@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    with open(CLASSIFIER_PATH, 'rb') as f:
        model, class_names = pickle.load(f)
    print("Custom Classifier, Successfully loaded")

    if request.method == 'POST':
        file = request.files['file']
        file_name = 'upload/%s' % file.filename
        file.save(file_name)

        name, prob = recognizer.recognition(
            file_name, model, class_names, sess, images_placeholder, 
            embeddings, phase_train_placeholder, pnet, rnet, onet)
        #prob = str(prob)

        return jsonify(name=name, prob=prob[0])

@app.route('/create-frame', methods=['POST'])
@cross_origin()
def upload_video_file():
    if request.method == 'POST':
        file = request.files['file']
        file_name = 'Video/%s' % file.filename
        file.save(file_name)
        name = file.filename
        name = str(name)
        name = name.split('.', 1)
        tmp = "created"
        mp4toimage.Create_frame(file_name, name[0])
        return jsonify(tmp=tmp)

@app.route('/align-face', methods=['POST'])
@cross_origin()
def upload_dataset():
    if request.method == 'POST':
        input_dir = "DataSet/FaceData/Raw" 
        nrof_images_total, nrof_successfully_aligned = align_dataset_mtcnn_update.align_face(input_dir, sess, pnet, rnet, onet)
        return jsonify(nrof_images_total=nrof_images_total, 
                        nrof_successfully_aligned=nrof_successfully_aligned)

@app.route('/classifier', methods=['POST'])
@cross_origin()
def classifier():
    if request.method == 'POST':
        tmp = "done classifier"
        classifier_update.main()
        return jsonify(tmp=tmp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')
