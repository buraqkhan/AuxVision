import time
import cv2
import numpy as np
import json
from datetime import datetime
import rect

rect_left = np.load('calibration/rectification_map_left.npy')
rect_right = np.load('calibration/rectification_map_right.npy')
und_left = np.load('calibration/undistortion_map_left.npy')   
und_right = np.load('calibration/undistortion_map_right.npy')

print ("You can press Q to quit this script!")
time.sleep (5)

# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100

# Use the whole image or a stripe for depth map?
useStripe = False
dm_colors_autotune = True
disp_max = -100000
disp_min = 10000


disparity = np.zeros((640, 480), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)


def stereo_depth_map(rectified_pair):
    global disp_max
    global disp_min
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    if (dm_colors_autotune):
        disp_max = max(local_max,disp_max)
        disp_min = min(local_min,disp_min)
        local_max = disp_max
        local_min = disp_min
        print(disp_max, disp_min)
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    #disparity_grayscale = (disparity+208)*(65535.0/1000.0) # test for jumping colors prevention 
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    cv2.imshow("Image", disparity_color)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();
    return disparity_color

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)


load_map_settings ("3dmap_set.txt")

cap_right = cv2.VideoCapture(4)                    
cap_left =  cv2.VideoCapture(2)

while(cap_right.isOpened() and cap_left.isOpened()):
    _, frame_right = cap_right.read()
    _, frame_left = cap_left.read()
    imgL = cv2.remap(frame_right, rect.und_left, rect.rect_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(frame_left, rect.und_right, rect.rect_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    imgL_new = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_new = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    rectified_pair = (imgL_new, imgR_new)
    disparity = stereo_depth_map(rectified_pair)
