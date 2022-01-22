import cv2 as cv
import hand_detections as h_md
from pynput.mouse import Button, Controller

#Define webcam's resolution
########################
wCam, hCam = 640, 480
########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = h_md.HandDetector(maxHands=1,detectionCon=0.8) 
mouse = Controller()

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
    #Find indexes
    lmList = detector.find_index(draw=False)
    
    #Interact
    ####################
    if lmList:
        # 1.Move mouse
        x,y = detector.get_palm(lmList)
        x, y = detector.calculate_ratio(img, x, y)
        mouse.position = (x*1960,y*1080)
        # 2.Left click
        print(detector.isUp(lmList,detector.INDEX))
        # 3.Right click
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