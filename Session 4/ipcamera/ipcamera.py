import cv2

# cap = cv2.VideoCapture('https://10.0.0.238:8080/video') 
cap = cv2.VideoCapture(0) 

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()