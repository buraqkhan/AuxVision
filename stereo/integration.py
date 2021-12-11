import depth_video
import numpy as np

'''
One function handles finding bounding boxes and calling the next function.
The other function then maps the output onto the depth map.
The output of this depth map will then be sent to the TTS module.
Put sleep(10) in between function calls.
Use openmp for threading? Update: Openmp doesn't exist in python
                                  Use multiprocessing for different functions instead? 
'''

def getAvgDistance(x1, y1, x2, y2, name):
    bounding_box = depth_video.disparity[x1:x2, y1:y2]
    average = np.median(bounding_box)
    distance = 420 + (1.3 * average) + (0.00168 * average**2)
    return (name, distance)  


def getBoundingBoxes():
    # Call Image model here 
    pass

def getDistances(_bounding_boxes):
    # Map co-ordinates or center points to the depth map
    bounding_boxes = _bounding_boxes
    distances = []
    for values in bounding_boxes:
        distances.append(getAvgDistance(values[0], values[1], values[2], values[3], values[4]))

    return distances

def TTS():
    # TTS module
    pass

def main():
    while(True):
        depth_video.runDisparity()
        #bb = [(350, 230, 570, 440, "chair"), (150, 11, 330, 450, "person")]
        # getBoundingBoxes()
        #print(getDistances(bb))
        # TTS()

if __name__ == "__main__":
    main()
