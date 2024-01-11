import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Load the image.
img = cv.imread('faces.jpg')

# Convert the image to RGB.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Process the image and detect the face mesh.
results = face_mesh.process(img_rgb)

# Check if the face mesh is detected.
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Specific landmarks for left and right eyes.
        left_eye_indices = [33, 133, 160, 158, 153, 144, 145, 163, 7]
        right_eye_indices = [362, 385, 387, 386, 380, 374, 373, 390, 249]

        # Function to calculate the bounding box with a margin.
        def calculate_bounding_box(eye_indices, margin=5):  # Margin added around the eye
            eye_coords = [(int(face_landmarks.landmark[i].x * img.shape[1]), int(face_landmarks.landmark[i].y * img.shape[0])) for i in eye_indices]
            x_coords, y_coords = zip(*eye_coords)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            return (x_min - margin, y_min - margin, x_max - x_min + 2 * margin, y_max - y_min + 2 * margin)

        # Drawing bounding box for the left eye with a margin.
        x, y, w, h = calculate_bounding_box(left_eye_indices)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Drawing bounding box for the right eye with a margin.
        x, y, w, h = calculate_bounding_box(right_eye_indices)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image.
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Release resources.
face_mesh.close()


