import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxNoOfHands=2, detectConfidence=0.5, trackConfidence=0.5) -> None:
        self.mode = mode
        self.maxNoOfHands = maxNoOfHands
        self.detectConfidence = detectConfidence
        self.trackConfidence = trackConfidence
        self.previousTime = 0
        self.currentTime = 0
        self.fps = 0
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxNoOfHands,
                                        min_detection_confidence=self.detectConfidence,
                                        min_tracking_confidence=self.trackConfidence)
        self.drawHand = mp.solutions.drawing_utils

    def detectHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.drawHand.draw_landmarks(frame, handLandmark, self.mpHands.HAND_CONNECTIONS)
        
        return frame
    
    def findPosition(self, frame, handNo=0, draw=True, landmarkId=None):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lmark in enumerate(myHand.landmark):
                h, w, c = frame.shape
                xPos, yPos = int(w * lmark.x), int(h * lmark.y)
                landmarkList.append([id, xPos, yPos])
                if draw and id == landmarkId:
                    cv.circle(frame, (xPos, yPos), 10, (210, 100, 140), cv.FILLED)
            
        return landmarkList
    
    def showFpsOnScreen(self, frame):
        self.currentTime = time.time()
        self.fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime
        cv.putText(frame, f"FPS: {str(int(self.fps))}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return frame




#this is just a usage example like how can we use this module in different files
# def main():
#     cap = cv.VideoCapture(0)
    
    
#     detector = handDetector()
    
#     while True:
#         res, frame = cap.read()
#         frame = detector.detectHands(frame)
#         landmarkList = detector.findPosition(frame, landmarkId=8)
#         frame = detector.showFpsOnScreen(frame)
#         if len(landmarkList) != 0:
#             print(landmarkList[8])
        
        
        
        
#         cv.imshow("Video", frame)
#         cv.waitKey(1)

# if __name__ == "__main__":
#     main()
