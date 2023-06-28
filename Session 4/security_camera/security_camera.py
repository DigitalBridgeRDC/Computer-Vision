import cv2
import winsound

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()

    # get the absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)

    # convert to gray scale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # convert frame to blur image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # threshold
    _, thresh = cv2.threshold(blur, 20,255,cv2.THRESH_BINARY)

    # dilation
    dilated = cv2.dilate(thresh, None, iterations=3)

    # contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (255,128,0),3)
        # alert sound
        # winsound.Beep(500, 200)
        winsound.PlaySound('alert.wav',winsound.SND_ASYNC)
        # winsound.PlaySound('dive_dive_dive.wav', winsound.SND_ASYNC)
    # show the frame
    cv2.imshow('Security Cam', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()