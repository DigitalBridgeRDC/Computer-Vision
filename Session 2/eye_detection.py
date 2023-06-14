import cv2

# Load the Haar cascades for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Read the input image
img = cv2.imread('eyeGaze.jpg')

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect eyes in the input image using the Haar cascades
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected eyes in the original image
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image with the detected eyes
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
