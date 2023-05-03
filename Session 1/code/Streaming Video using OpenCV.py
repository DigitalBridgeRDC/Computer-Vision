import cv2
import urllib.request
import numpy as np

# Open a video stream
stream = urllib.request.urlopen('http://192.168.1.100:8080/video')

# Start reading frames from the video stream
while True:
    # Read a frame from the video stream
    img_bytes = stream.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video stream and destroy all windows
stream.release()
cv2.destroyAllWindows()
