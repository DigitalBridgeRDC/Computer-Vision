import cv2

# Read an image in color format
img_color = cv2.imread('images/leo_hr.jpg')
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
# img_resized = cv2.resize(img_color, (300,300))
# Convert the image to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

cv2.imshow("Output", img_color)
cv2.imshow("gray", img_gray)


cv2.waitKey(0)