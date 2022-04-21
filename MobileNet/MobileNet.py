import cv2
import time
import queue
import numpy as np
from threading import Thread
from queue import Queue

'''
    Loading MobileNet model and specifying the weights to be used.
'''
classNames= []
classFile = 'MobileNet/coco.names'
with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]
configPath = 'MobileNet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'MobileNet/frozen_inference_graph.pb'   

'''
    Thread responsible for detecting and identifying objects in 
    frames and adding them to the output queue.
'''
def InputFramesThread(inputQueue,outputQueue,model,thres):
    model[0] = cv2.dnn_DetectionModel(weightsPath,configPath)
    model[0].setInputSize(320,320)

    model[0].setInputScale(1.0/ 127.5)
    model[0].setInputMean((127.5, 127.5, 127.5))
    model[0].setInputSwapRB(True)
    while True:
        img = inputQueue.get()
        model[1], model[2], model[3] = model[0].detect(img,confThreshold=thres)
        outputQueue.put(img)

'''
    Function responsible for checking which objects have been recently
    detected and returning them.
'''
def getObjects(coordinatesList):
    objectsDetected = dict()
    for coordinate in coordinatesList:
        if len(coordinate) != 0:
            if coordinate[4] not in objectsDetected:
                objectsDetected[coordinate[4]] = 1
            else:
                objectsDetected[coordinate[4]] += 1
        
    mostFrequentObjects = []

    for key,value in objectsDetected.items():
        if value > 15:
            mostFrequentObjects.append(key)

    return mostFrequentObjects

'''
    Function responsible for finding coordinates of the recently identified 
    objects. After finding the coordinates, it returns the most recent location.
'''

def getObjectsCoordinates(coordinatesList):
    recentCoordinates = []
    objectCoordinates = []
    mostFrequentObjects = getObjects(coordinatesList)
    for coordinate in coordinatesList:
        if len(coordinate) != 0:
            if coordinate[4] in mostFrequentObjects:
                recentCoordinates.append(coordinate)
    for object in mostFrequentObjects:
        for coordinate in reversed(recentCoordinates):
            if coordinate[4] == object:
                objectCoordinates.append(coordinate)
                break
    coordinatesList.clear()

    return objectCoordinates

'''
    Main driver function responsible for getting bounding boxes 
    which contain the object along with corresponding coordinates of
    that object.
'''

def getCoordinates(frame_right):
    thres = 0.6
    count = 0
    inputQueue = Queue(2)
    outputQueue = Queue()
    coordinates = []
    objects = []
    net = None
    classIds = None
    confs = None
    bbox = None
    model = []
    model.append(net)
    model.append(classIds)
    model.append(confs)
    model.append(bbox)
    inputThread = Thread(target=InputFramesThread, args= (inputQueue,outputQueue,model,thres))
    inputThread.daemon = True
    inputThread.start()
    timeout = 0
    while True:
        img = frame_right
        inputQueue.put(img)
        if outputQueue.empty():
            pass
        else:
            tempImg = outputQueue.get()
            
            if len(model[1]) != 0:
                for classId, confidence,box in zip(model[1].flatten(),model[2].flatten(),model[3]):  
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    coordinates.append(tuple((box[0],box[1],box[2]+box[0],box[3]+box[1],classNames[classId-1])))
                    
                    if (len(coordinates) % 60 == 0):
                        objects = getObjectsCoordinates(coordinates)
                        print(objects)
                        count = count + 1
                        if count == 3:
                            timeout = 0
                            cv2.imshow('Output',img)
                            while timeout < 350:
                                timeout = timeout + 1
                            return objects
            timeout = timeout + 1
            if timeout == 250:
                break
            # cv2.imshow('Output',img)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     return objects