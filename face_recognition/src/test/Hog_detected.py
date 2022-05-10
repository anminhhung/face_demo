import  dlib
import cv2
import numpy as np

image = cv2.imread("DataSet/FaceData/LeTuTuan1.jpg")

detector = dlib.get_frontal_face_detector()
facerects = detector(image, 0)

for facerect in facerects:
    x1 = facerect.left()
    y1 = facerect.top()
    x2 = facerect.right()
    y2 = facerect.bottom()

bounding_boxes = np.array([np.array([x1,y1,x2,y2])])
print(bounding_boxes)