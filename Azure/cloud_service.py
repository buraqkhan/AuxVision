from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
from PIL import Image

import os
import sys
import time
import cv2


with open("key.txt","r") as keyfile:
    subscription_key=keyfile.readline()

endpoint = "https://auxvision.cognitiveservices.azure.com/"
images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),"images")

computervision_client = ComputerVisionClient(endpoint,CognitiveServicesCredentials(subscription_key))

local_image_path_objects = os.path.join(images_folder,"pic.jpg")
# with open(local_image_path_objects,'rb') as f:
    #contents = f.read()
local_image_objects = open(local_image_path_objects,"rb")

detect_objects_results_local = computervision_client.detect_objects_in_stream(local_image_objects)
myimg = cv2.imread(local_image_path_objects,cv2.IMREAD_COLOR)
print("Detecting objects in image:")
if len(detect_objects_results_local.objects) == 0:
    print("No objects detected.")
else:
    for object in detect_objects_results_local.objects:
        # print(object)
        print(object.object_property)
        print("object at location {}, {}, {}, {}".format( \
        object.rectangle.x, object.rectangle.x + object.rectangle.w, \
        object.rectangle.y, object.rectangle.y + object.rectangle.h))
        x1,y1,x2,y2 = object.rectangle.x, object.rectangle.y, object.rectangle.x + object.rectangle.w, object.rectangle.y + object.rectangle.h
        conf = object.confidence
        label = object.object_property
        
        cv2.rectangle(myimg,(x1,y1),(x2,y2),(0,255,0),6)
        labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        cv2.putText(myimg,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
cv2.imshow('sample image',myimg)
cv2.waitKey(0)