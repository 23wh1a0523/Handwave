#Gen AI Hackathon 
#Team 70
#Team Lead:B.NISSI APEKSHA(23WH1A0523)
#Member1:P.TEENA PRANATHI(23WH1A0508)
#Member2:B.SAI DEVI SRI(23WH1A0513)
#Member3:P.KEERTHI(23WH1A0534)

import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

# Initialize webcam
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8, trackCon=0.8)

# Audio Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]  # -63.5 dB
maxVol = volRange[1]  # 0 dB
vol = 0
volBar = 400
volPer = 0

# Gesture Recognition Parameters
tipIds = [4, 8, 12, 16, 20]
mode = ""
active = 0

pyautogui.FAILSAFE = False
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers = []

    if len(lmList) >= 21:  # Ensure at least 21 landmarks (full hand detected)
        # Thumb
        fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)

        # 4 Fingers
        for id in range(1, 5):
            fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

        # Mode Switching
        if fingers == [0, 0, 0, 0, 0] and active == 0:
            mode = "Neutral"
        elif (fingers == [0, 1, 0, 0, 0] or fingers == [0, 1, 1, 0, 0]) and active == 0:
            mode = "Scroll"
            active = 1
        elif fingers == [1, 1, 0, 0, 0] and active == 0:
            mode = "Volume"
            active = 1
        elif fingers == [1, 1, 1, 1, 1] and active == 0:
            mode = "Cursor"
            active = 1

    # Scroll Control
    if mode == "Scroll":
        if fingers == [0, 1, 0, 0, 0]:
            pyautogui.scroll(300)
        elif fingers == [0, 1, 1, 0, 0]:
            pyautogui.scroll(-300)
        elif fingers == [0, 0, 0, 0, 0]:
            active = 0
            mode = "Neutral"

    # Volume Control
    if mode == "Volume" and len(lmList) >= 9:
        if fingers[-1] == 1:
            active = 0
            mode = "Neutral"
        else:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [50, 200], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            volPer = np.interp(vol, [minVol, maxVol], [0, 100])

    # Cursor Control
    if mode == "Cursor" and len(lmList) >= 9:
        if fingers[1:] == [0, 0, 0, 0]:  # Excluding thumb
            active = 0
            mode = "Neutral"
        else:
            x1, y1 = lmList[8][1], lmList[8][2]
            screen_w, screen_h = pyautogui.size()
            X = np.interp(x1, [50, 590], [0, screen_w])
            Y = np.interp(y1, [50, 430], [0, screen_h])
            pyautogui.moveTo(int(X), int(Y))
            if fingers[0] == 0:
                pyautogui.click()

    # Display Mode on Screen
    cv2.putText(img, f"Mode: {mode}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
