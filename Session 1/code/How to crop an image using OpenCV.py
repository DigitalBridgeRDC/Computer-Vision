import cv2

# Read an image
img = cv2.imread('images/vegetables.jpg')

# Define the region of interest (ROI) as a rectangle
x, y, w, h = 50, 250, 300, 300  # ROI coordinates and size
roi = img[y:y+h, x:x+w]

# Display the cropped image
cv2.imshow('Cropped Image', roi)
cv2.imshow('Original', img)
cv2.waitKey(0)
