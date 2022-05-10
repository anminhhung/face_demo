import cv2 
import time
import numpy as np
import pickle
import tensorflow as tf 

from utils.draw_image import draw_list_bbox_maxmin, draw_bbox_maxmin

from src.detect.dnn.detect_dnn import detect_face_ssd
from src.tracking.sort_tracking import * 
import face_recognition.src.facenet as facenet 
import face_recognition.src.align.detect_face as align_detect_face 


# Model
PROTOTXT = 'models/dnn/deploy.prototxt.txt'
MODEL = 'models/dnn/res10_300x300_ssd_iter_140000.caffemodel'
NET_DNN = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# setup sort tracking
SORT_TRACKER = Sort()

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'face_recognition/models/facemodel.pkl'
FACENET_MODEL_PATH = 'face_recognition/models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

def run_tracking():
    cap = cv2.VideoCapture(0)
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            while True:
                start_time = time.time()

                ret, frame = cap.read()

                ########## MAIN #################
                try:
                    list_face, list_score, list_classes = detect_face_ssd(frame, NET_DNN)

                    if len(list_face) > 0:
                        list_face = np.array(list_face)
                        # update SORT
                        track_bbs_ids = SORT_TRACKER.update(list_face)
                        # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                        for track in track_bbs_ids:
                            bbox = track[:4]

                            x = int(bbox[0])
                            y = int(bbox[1])
                            w = int(bbox[2]) - int(bbox[0])
                            h = int(bbox[3]) - int(bbox[1])

                            crop_image = frame[y:y+h, x:x+w]
                            scaled = cv2.resize(crop_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)

                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)

                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]

                            if best_class_probabilities > 0.9:
                                name = class_names[best_class_indices[0]]
                                frame = draw_bbox_maxmin(frame, (x, y, x+w, y+h), True, name)
                            else:
                                name = "Unknown"
                                frame = draw_bbox_maxmin(frame, (x, y, x+w, y+h), True, name)

                        # cal fps
                        fps = round(1.0 / (time.time() - start_time), 2)
                        print("fps: ", fps)

                except Exception as e:
                    print("Error: ", e)
                    pass
                #################################

                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    run_tracking()