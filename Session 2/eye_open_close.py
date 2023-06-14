import cv2
from scipy.spatial import distance as dist
import numpy as np

# Load the Haar cascades for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the facial landmark detector
facial_landmark_detector = cv2.dnn.readNetFromTensorflow('face_landmark_68_points.pb')

# Define the threshold for detecting closed eyes
eye_aspect_ratio_threshold = 0.25

# Define the number of consecutive frames for which the eyes must be closed to trigger an alert
consecutive_frames_threshold = 10

# Initialize the frame counter and consecutive frames counter
frame_counter = 0
consecutive_frames = 0

# Read the input video
cap = cv2.VideoCapture(0) # cap = cv2.VideoCapture('video.mp4')

# Loop over the frames in the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the input frame using the Haar cascades
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected eyes and extract the eye regions
    for (x, y, w, h) in eyes:
        eye_region = gray[y:y + h, x:x + w]

        # Detect the facial landmarks for the eye region using the facial landmark detector
        blob = cv2.dnn.blobFromImage(eye_region, scalefactor=1.0, size=(60, 60), mean=(0, 0, 0), swapRB=True, crop=False)
        facial_landmark_detector.setInput(blob)
        landmarks = facial_landmark_detector.forward()[0]
        landmarks = landmarks.reshape((68, 2))

        # Calculate the Eye Aspect Ratio (EAR) for the eye region
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = dist.euclidean(left_eye[1], left_eye[5]) / (2 * dist.euclidean(left_eye[0], left_eye[3]))
        right_ear = dist.euclidean(right_eye[1], right_eye[5]) / (2 * dist.euclidean(right_eye[0], right_eye[3]))
        ear = (left_ear + right_ear) / 2

        # Check if the eye is closed based on the EAR and the threshold
        if ear < eye_aspect_ratio_threshold:
            consecutive_frames += 1
        else:
            consecutive_frames = 0

        # Draw a label with the EAR and the eye state on the input frame
        cv2.putText(frame, 'EAR: {:.2f}'.format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if consecutive_frames >= consecutive_frames_threshold:
            cv2.putText(frame, 'Eyes closed!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Increment the frame counter
    frame_counter += 1

    # Display the output frame
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
