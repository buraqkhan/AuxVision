import copy
from stereo import depth_video
from TTS import tts
import numpy as np
from MobileNet.MobileNet import getCoordinates
import cv2
import socket
import struct
import pickle
from threading import Thread
from queue import Queue
from time import sleep
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

# def getAvgDistance(x1, y1, x2, y2, name):
    # print(x1, x2, y1, y2)
    # bounding_box = depth_video.disparity[x1:x2, y1:y2]
    # mean = np.mean(bounding_box)
    # std_dev = np.std(bounding_box)
    # print(std_dev)
    # dist_mean = abs(bounding_box - mean)
    # max_dev = 1
    # no_outliers = dist_mean < max_dev * (std_dev/2)
    # bounding_box = bounding_box[no_outliers]
    # average = np.mean(bounding_box)
    # mid_x = (x1 + x2)/2
    # mid_y = (y1 + y2)/2
    # average = depth_video.disparity[int(mid_x), int(mid_y)]
    # print(average)
    # distance = 420 + (1.3 * average) + (0.00168 * average**2)
    # return (name, distance)
''' =============================================================================   '''
HOST = "192.168.100.70"  # Standard loopback interface address (localhost)
PORT = 65431  # Port to listen on (non-privileged ports are > 1023)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
def runServer():
    data = b""
    payload_size = struct.calcsize(">L")
    img_received = "right"
    print("Server is now running!")
    print("I am now trying to receive image...")
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L",packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data,fix_imports=True,encoding="bytes")
    frame = copy.deepcopy(cv2.imdecode(frame,cv2.IMREAD_COLOR))
    print("Image received from " , img_received , " camera.")
    conn.sendall(bytes(img_received,'utf-8'))
    if img_received == "left":
        img_received = "right"
    elif img_received == "right":
        img_received = "left"

    my_image = None
    my_image = copy.deepcopy(frame)
    return my_image

def getAvgDistance(x1, y1, x2, y2, name):
    mid_x = int((x1 + x2)/2)
    mid_y = int((y1 + y2)/2)
    bounding_box = depth_video.disparity[mid_x-50:mid_x+50, mid_y-50:mid_y+50]
    average = np.mean(bounding_box)
    distance = 420 + (1.3 * average) + (0.00168 * average**2)
    return (name, distance)

def getBoundingBoxes(frame_right):
    # Call Image model here 
    return getCoordinates(frame_right)

def getDistances(_bounding_boxes):
    # Map co-ordinates or center points to the depth map
    bounding_boxes = _bounding_boxes
    distances = []
    for values in bounding_boxes:
        distances.append(getAvgDistance(values[0], values[1], values[2], values[3], values[4]))

    return distances

def TTS(boundingBoxes, distances):
    messages = []
    for i in range(len(boundingBoxes)):
        messages.append(tts.tts_object_location(boundingBoxes[i], distances[i][1]))
    return messages

def main():
    while(True):
        frame_right = runServer()
        frame_left = runServer()
        print("Frames have been gotten!")
            
        boundingBoxes = getBoundingBoxes(frame_right)
        if boundingBoxes is None:
            print("No object found!") 
            conn.sendall(bytes("No objects found,",'utf-8')) 
            continue
        print("Bounding boxes have been found!")
        print(boundingBoxes)

        depth_video.runDisparity(frame_right,frame_left)
        print("Depth has been done!")

        distances = getDistances(boundingBoxes)
        print("Distances have been found!")
        print(distances)

        feedback = ""
        messages = TTS(boundingBoxes, distances)
        for message in messages: 
            feedback += message
        print(messages)
        if len(messages) >= 1:
            conn.sendall(bytes(feedback,'utf-8'))
        else: 
            print("No object found!") 
            conn.sendall(bytes("No objects found,",'utf-8'))  
        print("TTS is done!")

        cv2.destroyAllWindows()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    conn.sendall(bytes("end",'UTF-8'))
    s.close()

if __name__ == "__main__":
    main()
