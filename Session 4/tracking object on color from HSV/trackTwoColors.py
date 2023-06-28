import numpy as np
import cv2
print(cv2.__version__)

def onTrack1(val):
    global hueLow1
    hueLow1 = val
    print('Hue Low 1', hueLow1)

def onTrack2(val):
    global hueHigh1
    hueHigh1 = val
    print('Hue High 1',hueHigh1)

def onTrack7(val):
    global hueLow2
    hueLow2 = val
    print('Hue Low 2', hueLow2)

def onTrack8(val):
    global hueHigh2
    hueHigh2 = val
    print('Hue High 2',hueHigh2)

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
height=480

#set camera properties
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# track bars
cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker',width, 0)
cv2.resizeWindow("myTracker", 320, height)

hueHigh1 = 135
hueLow1 = 90
satHigh = 255
satLow = 92
valHigh = 255
valLow = 189

hueHigh2 = 45
hueLow2 = 0

# Blue masking tape hueHigh = 135, hueLow = 90, satHigh = 255, satLow = 92, valHigh = 255, valLow = 189
# Red hueHigh2 = 45, hueLow2 = 0

cv2.createTrackbar('Hue Low 1', 'myTracker', hueLow1,179, onTrack1)
cv2.createTrackbar('Sat Low', 'myTracker', satLow,255, onTrack3)
cv2.createTrackbar('Value Low', 'myTracker', valLow,255, onTrack5)
cv2.createTrackbar('Hue High 1', 'myTracker', hueHigh1,179, onTrack2)
cv2.createTrackbar('Sat High', 'myTracker', satHigh,255, onTrack4)
cv2.createTrackbar('Value High', 'myTracker', valHigh,255, onTrack6)
cv2.createTrackbar('Hue Low 2', 'myTracker', hueLow2,179, onTrack7)
cv2.createTrackbar('Hue High 2', 'myTracker', hueHigh2,179, onTrack8)


while True:
    ret, frame = cam.read()
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerBound1 = np.array([hueLow1, satLow, valLow])
    upperBound1 = np.array([hueHigh1, satHigh, valHigh])

    lowerBound2 = np.array([hueLow2, satLow, valLow])
    upperBound2 = np.array([hueHigh2, satHigh, valHigh])

    myMask1 = cv2.inRange(frameHSV,lowerBound1, upperBound1)
    myMaskSmall1 = cv2.resize(myMask1, (int(width/2), int(height/2)))

    myMask2 = cv2.inRange(frameHSV,lowerBound2, upperBound2)
    myMaskSmall2 = cv2.resize(myMask2, (int(width/2), int(height/2)))

    myMaskComp = myMask1 | myMask2 #composite also: myMaskComp = cv2.add(myMask1,myMask2)

    myObject = cv2.bitwise_or(frame,frame, myMaskComp, mask=myMaskComp)
    myObjectSmall = cv2.resize(myObject, (int(width/2), int(height/2)))

    cv2.imshow("my Webcam", frame)
    cv2.moveWindow("my Webcam", 0,0)

    cv2.imshow("my Object", myObjectSmall)
    cv2.moveWindow("my Object", 0,height)

    cv2.imshow("my Mask1", myMaskSmall1)
    cv2.moveWindow("my Mask1", int(width/2),height)

    cv2.imshow("my Mask2", myMaskSmall2)
    cv2.moveWindow("my Mask2", int(width/2)*2,height)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()