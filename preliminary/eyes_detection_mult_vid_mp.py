import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)

# Define a margin for the bounding box.
margin = 2  # You can adjust this value as needed.

# Open the video file or capture device.
# cap = cv.VideoCapture('video2.mp4')  # Update the path to the video file or use 0 for webcam
cap = cv.VideoCapture(0)  # Update the path to the video file or use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame and detect the face mesh.
    results = face_mesh.process(frame_rgb)

    # Check if the face mesh is detected.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Identifying the landmarks for the left and right eye.
            left_eye_indices = [33, 133, 160, 158, 153, 144, 145, 163, 7]
            right_eye_indices = [362, 385, 387, 386, 380, 374, 373, 390, 249]

            # Function to calculate the bounding box with margin for eye indices.
            def calculate_bounding_box(eye_indices):
                eye_coords = [(int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0])) for i in eye_indices]
                x_coords, y_coords = zip(*eye_coords)
                x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
                y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
                return (x_min, y_min, x_max - x_min, y_max - y_min)

            # Drawing bounding box for the left eye.
            x, y, w, h = calculate_bounding_box(left_eye_indices)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Drawing bounding box for the right eye.
            x, y, w, h = calculate_bounding_box(right_eye_indices)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame.
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv.destroyAllWindows()
face_mesh.close()
