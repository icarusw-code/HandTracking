from cvzone.HandTrackingModule import HandDetector
import cv2
import socket
import math

#웹캠
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
success, img = cap.read()

# Parameters
h, w, _ = img.shape

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    
    # 데이터 초기화 
    data = []

    if hands:
        # Hand 1 첫번째 손을 가져옴
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points (x, y, z) * 21
        x1, x2, y1, y2 = 0, 0, 0, 0
        for lm in lmList:
            if lm[0] == 5:
                indexMid = lm[2]
                x1, y1 = lm[1], lm[2]
            elif lm[0] == 17:
                x2, y2 = lm[1], lm[2]
            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            data.extend([lm[0], h - lm[1], lm[2] + distance])

        sock.sendto(str.encode(str(data)), serverAddressPort)
        print(data)
    
    # Display
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) # 크기 작게 변환
    cv2.imshow("Image", img)
    cv2.waitKey(1)