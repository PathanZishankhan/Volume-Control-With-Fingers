import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as ht
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


camWidth, camHeight = 640, 480




devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()



minVol = volRange[0]
maxVol = volRange[1]





handTracker = ht.handDetector(detectConfidence=0.6)

cap = cv.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)


while True:
    res, frame = cap.read()
    
    frame = handTracker.detectHands(frame)
    frame = handTracker.showFpsOnScreen(frame)
    lmarkList = handTracker.findPosition(frame, landmarkId=4)
    
    
    if len(lmarkList) != 0:
        # print(lmarkList[4], lmarkList[8])
        x1, y1 = lmarkList[4][1], lmarkList[4][2]
        x2, y2 = lmarkList[8][1], lmarkList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        
        cv.circle(frame, (x1, y1), 15, (200,130,180), cv.FILLED)
        cv.circle(frame, (x2, y2), 15, (200,130,180), cv.FILLED)
        cv.line(frame, (x1,y1), (x2,y2), (180,200,130), 4)
        cv.circle(frame, (cx, cy), 10, (140,60,80), cv.FILLED)
        
        
        lineLen = math.hypot(x2 - x1, y2 - y1)
        # print(lineLen)
        
        
        # hand range 250:25
        # vol range = -65.25:0.00
        vol = np.interp(lineLen, [25, 230], [minVol, maxVol])
        volBar = np.interp(lineLen, [50, 300], [450, 400])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        
        if lineLen<20:
            cv.circle(frame, (cx, cy), 10, (0,255,0), cv.FILLED, 5)
    
    #cv.putText(frame, str(int(fps)), (15,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
    cv.imshow("Video",frame)
    cv.waitKey(1)