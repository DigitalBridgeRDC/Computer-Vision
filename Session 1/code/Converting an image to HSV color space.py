import cv2

# Read an image in color format
img_color = cv2.imread('images/leo.jpg')

# Convert the image to HSV color space
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

cv2.imshow("Original GBR", img_color)
cv2.imshow("Output HSV", img_hsv)

cv2.waitKey(0)
