import cv2 as cv
import math
import mediapipe as mp

class HandDetector():
    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    PINKY = 20

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
        self.img = img
        return img

    def find_index(self, indexNum=0, draw=True):
        lmList = []
        img = self.img
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[indexNum]

            for id, lm in enumerate(hand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 5, (255,0,255), cv.FILLED)   
        return lmList

    def get_palm(self, lmList, draw=True):
        if lmList:
            #Get the coordinate of the palm
            ####
            x1, y1 = lmList[0][1], lmList[0][2]
            x2, y2 = lmList[9][1], lmList[9][2]
            x = (x1 + x2)//2
            y = (y1 + y2)//2
            if draw:      
               cv.circle(self.img, (x,y), 3, (255,255,0), cv.FILLED)
            return x,y
        else: 
            return None

    def isUp(self, lmList, finger: int):
        if finger == self.THUMB:
            x1, y1 = lmList[self.THUMB][1], lmList[self.THUMB][2]
            x2, y2 = lmList[self.THUMB-1][1], lmList[self.THUMB-1][2]
            x3, y3 = lmList[self.THUMB-3][1], lmList[self.THUMB-3][2]
            return self.isVectorStraight((x2-x3,y2-y3), (x1-x3, y1-y3))
        elif finger == self.INDEX:
            x1, y1 = lmList[self.INDEX][1], lmList[self.INDEX][2]
            x2, y2 = lmList[self.INDEX-2][1], lmList[self.INDEX-2][2]
            x3, y3 = lmList[self.INDEX-3][1], lmList[self.INDEX-3][2]
            return self.isVectorStraight((x2-x3,y2-y3), (x1-x3, y1-y3))
        elif finger == self.MIDDLE:
            x1, y1 = lmList[self.MIDDLE][1], lmList[self.MIDDLE][2]
            x2, y2 = lmList[self.MIDDLE-1][1], lmList[self.MIDDLE-1][2]
            x3, y3 = lmList[self.MIDDLE-3][1], lmList[self.MIDDLE-3][2]
            return self.isVectorStraight((x2-x3,y2-y3), (x1-x3, y1-y3))
        elif finger == self.RING:
            x1, y1 = lmList[self.RING][1], lmList[self.RING][2]
            x2, y2 = lmList[self.RING-1][1], lmList[self.RING-1][2]
            x3, y3 = lmList[self.RING-3][1], lmList[self.RING-3][2]
            return self.isVectorStraight((x2-x3,y2-y3), (x1-x3, y1-y3))
        elif finger == self.PINKY:
            x1, y1 = lmList[self.PINKY][1], lmList[self.PINKY][2]
            x2, y2 = lmList[self.PINKY-1][1], lmList[self.PINKY-1][2]
            x3, y3 = lmList[self.PINKY-3][1], lmList[self.PINKY-3][2]
            return self.isVectorStraight((x2-x3,y2-y3), (x1-x3, y1-y3))

    def isStraight(self, vec1, vec2, confident=1):
        x1, y1 = vec1
        x2, y2 = vec2
        deno = x1*y2
        if deno == 0:
            deno += 10^-6
        ratio = (y1*x2)/deno
        return False if ratio < 0 or ratio > confident else True

    def isTip(self, lmList, finger: int, confident=50):
        x1, y1 = lmList[self.THUMB][1], lmList[self.THUMB][2]
        x2, y2 = 0, 0
        if finger == self.INDEX:
            x2, y2 = lmList[self.INDEX][1], lmList[self.INDEX][2]
        elif finger == self.MIDDLE:
            x2, y2 = lmList[self.MIDDLE][1], lmList[self.MIDDLE][2]
        elif finger == self.RING:
            x2, y2 = lmList[self.RING][1], lmList[self.RING][2]
        elif finger == self.PINKY:
            x2, y2 = lmList[self.PINKY][1], lmList[self.PINKY][2]
        length = math.hypot(x1-x2, y1-y2)
        return True if length <= confident else False
        
    def calculate_ratio(self, img, x, y):
        h, w = img.shape[:2]
        ratx = x/w
        raty = y/h
        return ratx, raty