import cv2

# Load an image
img = cv2.imread('images/vegetables.jpg')

# Create a window and display the image
cv2.imshow('Window Name', img)

# Wait for any key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
