import cv2

# Read an image
img = cv2.imread('images/db.jpg')

# 1560 x 729
# Resize the image to a specific size
resized = cv2.resize(img, (int(1560/2), int(729/2)))

# Display the resized image
cv2.imshow('Resized Image', resized)
cv2.imshow('Orignal Image', img)
cv2.waitKey(0)
