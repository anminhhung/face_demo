# Import libraries
import os
import cv2
import numpy as np

# Define paths

prototxt_path = 'Models/deploy.prototxt'
caffemodel_path = 'Models/weights.caffemodel'

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


def Detected(image, model):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    return detections

def Cropped(image, model, bounding_box):
    (h, w) = image.shape[:2]
    Max = bounding_box[0, 0, 0, 2]
    out = "False"
    # Identify each face
    for i in range(0, bounding_box.shape[2]):
        box = bounding_box[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = bounding_box[0, 0, i, 2]
        
        # If confidence > 0.8, save it as a separate file
        if confidence >= Max:
            frame = image[startY:endY, startX:endX]
            Max = confidence
        
    if Max < 0.8:
        return out
    else:
        return frame

