import numpy as np
import cv2
print(cv2.__version__)

def onTrack1(val):
    global hueLow
    hueLow = val
    print('Hue Low', hueLow)

def onTrack2(val):
    global hueHigh
    hueHigh = val
    print('Hue High',hueHigh)

def onTrack3(val):
    global satLow
    satLow = val
    print('Sat Low',satLow)

def onTrack4(val):
    global satHigh
    satHigh = val
    print('Sat High',satHigh)

def onTrack5(val):
    global valLow
    valLow = val
    print('Val Low',valLow)

def onTrack6(val):
    global valHigh
    valHigh = val
    print('Val High',valHigh)

width=640
height=360

#set camera properties
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# track bars
cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker',width, 0)

hueHigh = 135
hueLow = 90
satHigh = 255
satLow = 92
valHigh = 255
valLow = 189

# Blue masking tape hueHigh = 135, hueLow = 90, satHigh = 255, satLow = 92, valHigh = 255, valLow = 189

cv2.createTrackbar('Hue Low', 'myTracker', hueLow,179, onTrack1)
cv2.createTrackbar('Sat Low', 'myTracker', satLow,255, onTrack3)
cv2.createTrackbar('Value Low', 'myTracker', valLow,255, onTrack5)
cv2.createTrackbar('Hue High', 'myTracker', hueHigh,179, onTrack2)
cv2.createTrackbar('Sat High', 'myTracker', satHigh,255, onTrack4)
cv2.createTrackbar('Value High', 'myTracker', valHigh,255, onTrack6)


while True:
    ret, frame = cam.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerBound = np.array([hueLow, satLow, valLow])
    upperBound = np.array([hueHigh, satHigh, valHigh])

    myMask = cv2.inRange(frameHSV,lowerBound, upperBound)
    myMaskSmall = cv2.resize(myMask, (int(width/2), int(height/2)) )

    myObject = cv2.bitwise_and(frame,frame, myMask, mask=myMask)
    myObjectSmall = cv2.resize(myObject, (int(width/2), int(height/2)) )

    cv2.imshow("my Webcam", frame)
    cv2.moveWindow("my Webcam", 0,0)

    cv2.imshow("my Object", myObjectSmall)
    cv2.moveWindow("my Object", 0,height)

    cv2.imshow("my Mask", myMaskSmall)
    cv2.moveWindow("my Mask", int(width/2),height)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()