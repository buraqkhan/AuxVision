from stereo import depth_video
from TTS import tts
import numpy as np
from MobileNet.MobileNet import getCoordinates
import cv2

import sys
import os

sys.path.append(os.getcwd() + '/obj_detect_pi')

from obj_detect_pi.detect import tf_run

'''
    Main file that runs all the functionality.
    Driver Function (Main):
        1) getBoundingBoxes() called first, returns bounding boxes from the video feed.
        2) runDisparity() starts up the depth map, which returns depth values.
        3) getDistances() return the actual distance values of the objects detected.
        4) The output will then be sent to the TTS module using TTS(). 
'''

''' =========  Different Test Functions (Not relevant for this iteration) ===========  '''
# def getAvgDistance(x1, y1, x2, y2, name):
#     bounding_box = depth_video.disparity[x1:x2, y1:y2]
#     average = np.median(bounding_box)
#     distance = 420 + (1.3 * average) + (0.00168 * average**2)
#     return (name, distance)  
''' =============================================================================   '''
 
def getAvgDistance(x1, y1, x2, y2, name):
    mid_x = int((x1 + x2)/2)
    mid_y = int((y1 + y2)/2)
    bounding_box = depth_video.disparity[mid_x-50:mid_x+50, mid_y-50:mid_y+50]
    average = np.mean(bounding_box)
    distance = 691 + (-1.59 * average) + (0.00116 * average**2)
    return (name, distance)

def getBoundingBoxes(cap_right):
    # Call Image model here 
    return getCoordinates(cap_right)

def getDistances(_bounding_boxes):
    # Map co-ordinates or center points to the depth map
    bounding_boxes = _bounding_boxes
    distances = []
    for values in bounding_boxes:
        distances.append(getAvgDistance(values[0], values[1], values[2], values[3], values[4]))

    return distances

def TTS(boundingBoxes, distances):
    for i in range(len(boundingBoxes)):
        tts.tts_object_location(boundingBoxes[i], distances[i][1])

def main():
    cap_right = cv2.VideoCapture(0)                    
    # cap_left =  cv2.VideoCapture(4)
    # tf_run(cap_right)
    
    # while(True):
    #     boundingBoxes = tf_run(cap_right)
    #     print(boundingBoxes)
    #     depth_video.runDisparity(cap_right,cap_left)
    #     distances = getDistances(boundingBoxes)
    #     print(distances)

    #     # boundingBoxes = [(350, 230, 570, 440, "chair"), (150, 11, 330, 450, "person")]
    #     # distances = [('chair', 158), ('person', 230)]

    #     TTS(boundingBoxes, distances)
    #     break

if __name__ == "__main__":
    main()
