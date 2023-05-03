import cv2 # imports OpenCV

# Read an image in color format
img_color = cv2.imread('images/vegetables.jpg')

# Access the individual color channels
img_red = img_color[:, :, 2]  # red channel
img_green = img_color[:, :, 1]  # green channel
img_blue = img_color[:, :, 0]  # blue channel

cv2.imshow("original", img_color)
cv2.imshow("red", img_red)
cv2.imshow("green", img_green)
cv2.imshow("blue", img_blue)

cv2.waitKey(0)