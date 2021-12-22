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
    # model[0].setInputSize(640,480)

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
    # print(coordList[-5:])
    # RecentCoordinates = coordList[-20:]
    for coordinate in coordinatesList:
        # print(coordinate)
        if len(coordinate) != 0:
            if coordinate[4] not in objectsDetected:
                objectsDetected[coordinate[4]] = 1
            else:
                objectsDetected[coordinate[4]] += 1
        
    # objectsDetected = dict(sorted(objectsDetected.items(), key=lambda item: item[1]))
    mostFrequentObjects = []

    for key,value in objectsDetected.items():
        # print(key,value)
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
        # print(coordinate)
        if len(coordinate) != 0:
            if coordinate[4] in mostFrequentObjects:
                recentCoordinates.append(coordinate)
    for object in mostFrequentObjects:
        for coordinate in reversed(recentCoordinates):
            if coordinate[4] == object:
                objectCoordinates.append(coordinate)
                break
    coordinatesList.clear()
    # print(mostFrequentObjects)
    # print(objectCoordinates)

    return objectCoordinates

'''
    Main driver function responsible for getting bounding boxes 
    which contain the object along with corresponding coordinates of
    that object.
'''

def getCoordinates(cap_right):
    thres = 0.6
    # cap = cv2.VideoCapture(2)  
    #cap.set(3,1280)
    #cap.set(4,720)
    #cap.set(10,70) 
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
    while True:
        startTime = time.time()
        success,img = cap_right.read()
        # print(np.shape(img))
        # img = cv2.resize(img,(320,320))
        inputQueue.put(img)
        
        if outputQueue.empty():
            pass
        else:
            # print(type(model[0]))
            # print(type(model[1]))
            # print(type(model[2]))
            # print(type(model[3]))
            tempImg = outputQueue.get()
            if len(model[1]) != 0:
                # print("Num of BBox : ", len(model[3]))
                for classId, confidence,box in zip(model[1].flatten(),model[2].flatten(),model[3]):
                    # print(box)
                    
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    coordinates.append(tuple((box[0],box[1],box[2]+box[0],box[3]+box[1],classNames[classId-1])))
                    if (len(coordinates) % 60 == 0):
                        objects = getObjectsCoordinates(coordinates)
                        # print("60 coordinates found!")
                        print(objects)
                        # print("All objects printed")

                    #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            endTime = time.time()
            totalTime = endTime - startTime
            fps = 1 / totalTime
            # print("FPS: ", fps)
            cv2.putText(img, f'FPS: {int(fps)}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Output',320,320)
            # img = cv2.resize(img, (320,320))
            cv2.imshow('Output',img)
            # time.sleep(1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                    # cap_right.release()
                    # cv2.destroyAllWindows()
                    return objects

# coordinate = getCoordinates()