import cv2 as cv
import numpy as np

########################
wCam, hCam = 640, 480
########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while cap.isOpened():
    success, img = cap.read()

    cv.imshow("Cam", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()