import cv2 as cv
import numpy as np
import mediapipe as mp
import mouse

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5) -> None:
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHands, complexity, detectionCon, trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img):    
        #Flip
        img = cv.flip(img, 1)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        #Detect hand
        self.results = self.hands.process(rgb)

        #Draw connecting lines
        if self.results.multi_hand_landmarks:
            #Draw circles
            for handlandmark in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handlandmark, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_index(self, img, indexNum=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[indexNum]

            for id, lm in enumerate(hand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 5, (255,0,255), cv.FILLED)   
        return lmList

    def calculate_ratio(self, img, x, y):
        w, h = img.shape[:2]
        ratx = x/w
        raty = y/h
        return ratx, raty

detector = HandDetector()

if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_index(img)

        if lmList:
            id, x, y = lmList[8]
            x, y = detector.calculate_ratio(img, x, y)
            mouse.moveMouse((x*1960,y*1080))

        cv.imshow('Normal', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()