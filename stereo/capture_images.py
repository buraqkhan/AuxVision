import numpy as np
import cv2

vc_0 = cv2.VideoCapture(4)
vc_1 = cv2.VideoCapture(2)
# vc_1.set(3, 320)
# vc_1.set(4, 200)
# vc_0.set(3, 320)
# vc_0.set(4, 200)
num = 0

while True:
    ret0, left = vc_0.read()
    #left = cv2.flip(left, -1)
    ret1, right = vc_1.read()
    # right = cv2.resize(right, (320, 200), interpolation= cv2.INTER_AREA)

    if ret0:
        cv2.imshow("Left", left)
    
    if ret1:
        cv2.imshow("Right", right)
    
    k = cv2.waitKey(3)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('images/left_' + str(num) + '.ppm', left)
        cv2.imwrite('images/right_' + str(num) + '.ppm', right)
        print("saved")
        num += 1

vc_0.release()
vc_1.release()
cv2.destroyAllWindows()

# cam 1 is left
