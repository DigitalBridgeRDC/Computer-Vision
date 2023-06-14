import cv2
import numpy as np

# Load the Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the image to be processed
img = cv2.imread('more_right.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect eyes in the image
eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

# Iterate through the detected eyes
for (x,y,w,h) in eyes:
    # Estimate the gaze direction 
    eye_center = (x + w//2, y + h//2)
    gaze_direction = np.arctan2(eye_center[0] - img.shape[1]//2, eye_center[1] - img.shape[0]//2)
    
    # Draw the gaze direction on the image
    cv2.arrowedLine(img, tuple(eye_center), (img.shape[1]//2, img.shape[0]//2), (0,0,255), 2)
    
# Display the image
cv2.imshow('image', img)
cv2.moveWindow('image', 1000,800)
cv2.waitKey(0)
cv2.destroyAllWindows()
