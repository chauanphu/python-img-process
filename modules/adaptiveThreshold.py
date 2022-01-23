import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()

    blur = cv.bilateralFilter(img, 3, 15, 15)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    canny = cv.Canny(img, 100, 125)
    cv.imshow('Normal', img)
    cv.imshow('Canny', canny)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

