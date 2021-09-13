import cv2
import numpy as np 
# from stereovision.calibration import StereoCalibration
# from stereovision.stereo_cameras import CalibratedPair
# from stereovision.blockmatchers import StereoBM

# image_pair = [cv2.imread(image) for image in ["/test/left", "/test/right"]]

# block_matcher = StereoBM()   
# camera_pair = CalibratedPair(None, StereoCalibration(input_folder="/calibration/"), block_matcher)
# rectified_pair = camera_pair.calibration.rectify(image_pair)    

rect_left = np.load('calibration/rectification_map_left.npy')
rect_right = np.load('calibration/rectification_map_right.npy')
und_left = np.load('calibration/undistortion_map_left.npy')   
und_right = np.load('calibration/undistortion_map_right.npy')

# left = cv2.imread("test/left/imageL0.png")
# right = cv2.imread("test/right/imageR0.png")

# # cv2.imshow("left",left)
# # cv2.imshow("right", right)
# # cv2.waitKey(0)

# newL = cv2.remap(left, und_left, rect_left, cv2.INTER_NEAREST)
# newR = cv2.remap(right, und_right, rect_right, cv2.INTER_NEAREST)

# cv2.imwrite("rect_left.png", newL)
# cv2.imwrite("rect_right.png", newR)

# test1 = np.load('calibration/cam_mats_left.npy')
# test2 = np.load('calibration/cam_mats_right.npy')

# print(test1)
# print(test2)

# cv2.imshow("left", newL)
# cv2.imshow("right", newR)
# cv2.waitKey(0)      
