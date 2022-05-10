import cv2
import numpy as np
import random
import os
import sys
from time import sleep
import facenet
from scipy import misc
import DNN_detected
import time

output_dir = "DataSet/FaceData/DNN_processed"
input_dir = "DataSet/FaceData/raw"
prototxt_path = 'Models/deploy.prototxt'
caffemodel_path = 'Models/weights.caffemodel'
random_order = True
detect_mutiple_faces = False
image_size = 160

def main():
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    sleep(random.random())    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(input_dir)

    # Add a random key to the filename to allow alignment using multiple processes
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_filename.txt')
    miss_image_file = os.path.join(output_dir, 'miss_image_file.txt')

    start = time.time()
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'. format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print("Unable to align '%s'" % image_path)
                            text_file.write("%s\n" % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]
                        bounding_boxes = DNN_detected.Detected(img, model)
                        tmp  = () #compare with with bounding_boxes
                        if bounding_boxes == tmp:
                            bounding_boxes = np.empty(0)

                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                if detect_mutiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                scaled = DNN_detected.Cropped(img, model, bounding_boxes)
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)

                                if detect_mutiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                if scaled == "False":
                                    print('Unable to align "%s"' % image_path)
                                    nrof_successfully_aligned -= 1
                                    with open(miss_image_file, "a") as f:
                                        f.write('%s\n' % (output_filename))
                                else:
                                    misc.imsave(output_filename_n, scaled)
                                    text_file.write('%s\n' % (output_filename_n))
                        else:
                            print('Unable to align "%s"' % image_path)
                            with open(miss_image_file, "a") as f:
                                f.write('%s\n' % (output_filename))
    end = time.time()
    print('Detection_neural_network:')
    print('time processed: ', end-start)
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)                           

main()