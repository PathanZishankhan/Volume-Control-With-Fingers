import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule


handTracker = HandTrackingModule.handDetector()

cap = cv.VideoCapture(0)
currentTime=0
previousTime=0
while True:
    res, frame = cap.read()
    
    frame = handTracker.detectHands(frame, draw=False)
    frame = handTracker.showFpsOnScreen(frame)
    handTracker.findPosition(frame, landmarkId=4)

    
    #cv.putText(frame, str(int(fps)), (15,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
    cv.imshow("Video",frame)
    cv.waitKey(1)