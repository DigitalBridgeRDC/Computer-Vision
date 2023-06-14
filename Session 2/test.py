import cv2
import numpy as np

# Load the pre-trained age estimation model
model = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'dex_chalearn_iccv2015.caffemodel')

# Load the input image
image = cv2.imread('image.jpg')

# Resize the image to the input size of the model
input_size = (256, 256)
resized_image = cv2.resize(image, input_size)

# Preprocess the image for the model
preprocessed_image = cv2.dnn.blobFromImage(resized_image, 1.0, input_size, (0, 0, 0), swapRB=True)

# Set the input of the model
model.setInput(preprocessed_image)

# Forward pass through the model
output = model.forward()

# Extract the predicted age from the output
age = int(output[0][0])

# Display the input image with the estimated age
cv2.putText(image, f"Age: {age}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
