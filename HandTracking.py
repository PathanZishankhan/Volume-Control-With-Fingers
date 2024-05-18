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
    handTracker.detectHands(frame)
    handTracker.findPosition(frame, landmarkId=4)
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    
    cv.putText(frame, str(int(fps)), (15,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
    cv.imshow("Video",frame)
    cv.waitKey(1)