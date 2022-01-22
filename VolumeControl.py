import cv2 as cv
import volume
import hand_detections as h_md
import math

#Define webcam's resolution
########################
wCam, hCam = 640, 480
########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = h_md.HandDetector(detectionCon=0.7) 

def cal_volumm(value=0, min=50, max=250):
    if value >= max: 
        percent = 100
    elif value <= min:
        percent = 0
    else:
        percent = int((value-min)/(max-min)*100)
    return percent

def activated(value=0, min=30, max=50):
    if value >= max: 
        return True
    elif value <= min:
        return False

# Main process
##################################
while cap.isOpened():
    success, img = cap.read()

    img = detector.find_hands(img)
    lmList = detector.find_index(img, draw=False)
    
    #Find indexes
    ####################
    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv.circle(img, (x1,y1), 3, (255,0,0), cv.FILLED)
        cv.circle(img, (x2,y2), 3, (255,0,0), cv.FILLED)
        x3, y3 = lmList[12][1], lmList[12][2]
        x4, y4 = lmList[16][1], lmList[16][2]
        cv.circle(img, (x3,y3), 3, (255,255,0), cv.FILLED)
        cv.circle(img, (x4,y4), 3, (255,255,0), cv.FILLED)
        

        cv.line(img, (x3,y3), (x4,y4), (255,255,0), 3)

        length1 = math.hypot(x1-x2, y1-y2)
        length2 = math.hypot(x3-x4, y3-y4)

        if activated(length2):
            cv.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
            vol = cal_volumm(length1)
            volume.set_volume(vol)
    ####################

    # Display camera
    cv.imshow("Cam", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
##################################

# Destroy all windows
########################
cap.release()
cv.destroyAllWindows()
########################