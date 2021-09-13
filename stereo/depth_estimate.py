import sys
import cv2
from matplotlib.transforms import Bbox
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import torch

# Function for stereo vision and depth estimation
import triangulation as tri
import rect

import time


model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained=True)  # or yolov5m, yolov5l, yolov5x, custom
model.classes = [67]

# Open both cameras
cap_right = cv2.VideoCapture(2)                    
cap_left =  cv2.VideoCapture(0)



# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps
B = 9

while(cap_right.isOpened() and cap_left.isOpened()):
    #print("hello")
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    frame_left = cv2.remap(frame_left, rect.und_left, rect.rect_left, cv2.INTER_NEAREST)
    frame_right = cv2.remap(frame_right, rect.und_right, rect.rect_right, cv2.INTER_NEAREST)

########################################################################################

    if not success_right or not success_left:                    
        break

    start = time.time()
    
    # Process the image and find faces
    results_right = model(frame_right)
    results_left = model(frame_left)


    ################## CALCULATING DEPTH #########################################################

    center_right = 0
    center_left = 0

    if results_right.xyxy[0].size() != torch.Size([0, 6]):        
        results_right.render()
        bBox = results_right.xyxy[0].tolist()
        h, w, c = frame_right.shape

        boundBox = int(bBox[0][0] * w), int(bBox[0][1] * h), int(bBox[0][2] * w), int(bBox[0][3] * h)

        center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        cv2.putText(frame_right, f'{int(bBox[0][4])}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


    if results_left.xyxy[0].size() != torch.Size([0, 6]):        
        results_left.render()
        bBox = results_left.xyxy[0].tolist()
        h, w, c = frame_left.shape

        boundBox = int(bBox[0][0] * w), int(bBox[0][1] * h), int(bBox[0][2] * w), int(bBox[0][3] * h)

        center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        cv2.putText(frame_left, f'{int(bBox[0][4])}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)





    # If no ball can be caught in one camera show text "TRACKING LOST"
    if results_right.xyxy[0].size() == torch.Size([0, 6]) or results_left.xyxy[0].size() == torch.Size([0, 6]):
        cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    else:
        # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
        # All formulas used to find depth is in video presentaion
        depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B)

        cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
        print("Depth: ", str(round(depth,1)))



    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    #print("FPS: ", fps)

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


    # Show the frames
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)


    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()