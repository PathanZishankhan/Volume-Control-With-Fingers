import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
drawHand = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0


while True:
    res, frame = cap.read()
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, lmark in enumerate(handLandmark.landmark):
                h, w, c = frame.shape
                
                xPos, yPos = int(w * lmark.x), int(h * lmark.y)
                
                if id == 0:
                    cv.circle(frame, (xPos, yPos), 10, (210, 100, 140), cv.FILLED)
            
            drawHand.draw_landmarks(frame, handLandmark, mpHands.HAND_CONNECTIONS)
            
            
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    
    cv.putText(frame, str(int(fps)), (15,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
    
    
    
    cv.imshow("Video",frame)
    cv.waitKey(1)