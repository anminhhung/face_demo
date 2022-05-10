import cv2
import numpy as np
import os
'''
    Input: Path of video.
    Output: 100 images(name: MSSV_i "i is ID images") in folder(name: MSSV) (folder MSSV in  Raw)
    flow: Get file MSSV.mp4 in CongVu's file
        -> run api
        -> MSSV.mp4 back up to folder video
        -> return "done"
'''

def Create_frame(path_video, MSSV):     
    cap = cv2.VideoCapture(path_video)
    path = "./DataSet/FaceData/Raw/"
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        if not os.path.exists(path + MSSV): #convert MSSV int to str 
            os.makedirs(path + MSSV)
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(currentFrame<100):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Saves image of the current frame in jpg file
        name = path + MSSV + '/' + MSSV + '_' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
