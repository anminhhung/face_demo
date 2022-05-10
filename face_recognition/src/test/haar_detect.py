import cv2
#image_size = 160


CASCADE_PATH = "Models/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
Image_path = "DataSet/FaceData/LeTuTuan1.jpg"
img = cv2.imread(Image_path)

def detect(img, face_cascade):
    if (img is None):
        print("Can't open image file")
        return 0

    faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    if (faces is None):
        print('Failed to detect face')
        return 0

    return faces
    
def cropped(img, faces, image_size):
    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (image_size, image_size))
    return lastimg

def generate(img):
    face_detected = detect(img, face_cascade)
    print(face_detected.shape)

generate(img)