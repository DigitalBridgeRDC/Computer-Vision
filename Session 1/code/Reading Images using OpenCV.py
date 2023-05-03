import cv2

# Load an image in color format
img_color = cv2.imread('path/to/image.jpg', cv2.IMREAD_COLOR)

# Load an image in grayscale format
img_gray = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)

# Load an image as is, including alpha channel if present
img_unchanged = cv2.imread('path/to/image.png', cv2.IMREAD_UNCHANGED)
