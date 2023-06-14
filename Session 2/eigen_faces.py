import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people

# Load the LFW dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Extract the face images and labels
faces = lfw_dataset.data
labels = lfw_dataset.target
num_faces, height, width = faces.shape[0], faces.shape[1], faces.shape[2]

# Compute the mean face and the centered faces
mean_face = np.mean(faces, axis=0)
centered_faces = faces - mean_face

# Compute the principal components using PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True)
pca.fit(centered_faces)
eigenfaces = pca.components_.reshape((n_components, height, width))

# Load a test image
test_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the test image to the same size as the training images
resized_test_image = cv2.resize(test_image, (width, height))

# Center the test image using the mean face
centered_test_image = resized_test_image - mean_face

# Project the test image onto the principal components
test_projection = pca.transform(centered_test_image.flatten().reshape(1, -1))

# Compute the Euclidean distances between the test projection and the training projections
distances = np.linalg.norm(pca.transform(centered_faces.reshape(num_faces, -1)) - test_projection, axis=1)

# Find the index of the closest matching face
min_index = np.argmin(distances)

# Display the input image and the closest matching face
cv2.imshow('input', test_image)
cv2.imshow('match', faces[min_index].reshape(height, width))
cv2.waitKey(0)
