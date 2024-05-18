import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

while True:
    res, frame = cap.read()
    
    cv.imshow("Video",frame)
    cv.waitKey(1)