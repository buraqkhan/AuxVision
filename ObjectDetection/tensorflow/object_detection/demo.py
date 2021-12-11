import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import queue
import threading
from threading import Thread
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from queue import Queue


# What model to download.
# Models can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

# Download Model
if not os.path.exists(os.path.join(os.getcwd(), MODEL_FILE)):
    print("Downloading model")
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())






# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# coordinates_list.append([ymin, ymax, xmin, xmax, box_to_display_str_map[box][0]])
            #make function, 5 last frames. coordinates + class name of each stored. On avg, which class came more than 2 times. 
            # return most frequent classes + their BB coordinates.

def getObjectsCoordinates(coordinatesList):
    recentCoordinates = []
    objectCoordinates = []
    mostFrequentObjects = getObjects(coordinatesList)
    for coordinate in coordinatesList:
        if len(coordinate) != 0:
            if coordinate[0][4] in mostFrequentObjects:
                recentCoordinates.append(coordinate)
    for object in mostFrequentObjects:
        for coordinate in reversed(recentCoordinates):
            if coordinate[0][4] == object:
                objectCoordinates.append(tuple(coordinate[0]))
                break
    coordinatesList.clear()
    # print(mostFrequentObjects)
    # print(objectCoordinates)

    return objectCoordinates

def getObjects(coordList):
    # if len(coordList) >= 20:
    objectsDetected = dict()
    # print(coordList[-5:])
    # RecentCoordinates = coordList[-20:]
    for coordinate in coordList:
        # print(coordinate)
        if len(coordinate) != 0:
            if coordinate[0][4] not in objectsDetected:
                objectsDetected[coordinate[0][4]] = 1
            else:
                objectsDetected[coordinate[0][4]] += 1
        
    # objectsDetected = dict(sorted(objectsDetected.items(), key=lambda item: item[1]))
    mostFrequentObjects = []

    for key,value in objectsDetected.items():
        # print(key,value)
        if value > 15:
            mostFrequentObjects.append(key)

    return mostFrequentObjects


        


def InputFrameThread(FrameQueue,DisplayQueue,coordinatesList):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # sess = tf.Session(graph=detection_graph)
        sess = tf.compat.v1.Session(graph=detection_graph) 
    with detection_graph.as_default():
        while True:
            image_np = FrameQueue.get()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)


            coordinatesList.append(vis_util.getCoordinates(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.6
                ))

            # print(np.size(coordinatesList))
            # for i in range(len(coordinatesList)):
            #     print(coordinatesList[i])

            # max_boxes_to_draw = boxes.shape[0]
            # min_score_thresh = 0.7
            # idx = 0
            # for i in range(min(max_boxes_to_draw,boxes.shape[0])):
            #    if scores[0][i] > min_score_thresh:
            #        class_id = int(classes[0][i])
                #    coordinates.append({
                #        "box":boxes[i],
                #        "class_name": category_index[class_id]["name"],
                #        "score":scores[i]
                #    })

            

            #for i in range(len(coordinates)):
             #   print(coordinates)

            

            DisplayQueue.put(image_np)
    

if __name__ == '__main__':
     # Define the video stream
    cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
    FrameQueue = Queue(2)
    DisplayQueue = Queue()
    coordinatesList = []
    mostFrequentObjects = []
    FrameThread = Thread(target = InputFrameThread, args=(FrameQueue,DisplayQueue,coordinatesList))
    FrameThread.daemon = True
    # ObjectsThread = Thread(target=getObjectsThread,arg=(coordinatesList))
    # ObjectsThread.daemon = True
    # ObjectsThread.start()
    FrameThread.start()
    # ObjectsThread.start()
    
    # Detection
    while True:
        startTime = time.time()
        ret, image_np = cap.read()
        image_np = cv2.resize(image_np, (640, 480))
        # print(np.shape(image_np))
        FrameQueue.put(image_np)
        if DisplayQueue.empty():
            pass
        else:
            image_np_display = DisplayQueue.get()
            endTime = time.time()
            totalTime = endTime - startTime
            fps = 1 / totalTime
            # print("FPS: ", fps)
            cv2.putText(image_np_display, f'FPS: {int(fps)}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('object detection', cv2.resize(image_np_display, (640, 480)))
            if len(coordinatesList) % 60 == 0:
                objectCoordinates = getObjectsCoordinates(coordinatesList)
                print(objectCoordinates)
                
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
                
