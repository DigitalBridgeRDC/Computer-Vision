import cv2
import numpy as np

# Load the pre-trained deep learning face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Read the input image
img = cv2.imread('input.jpg')

# Get the image dimensions and construct a blob from the image
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the deep learning face detection model to detect faces
net.setInput(blob)
detections = net.forward()

# Loop over the detections and draw rectangles around the detected faces in the original image
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the output image with the detected faces
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
