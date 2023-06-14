import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    for face in faces:
        # Detect landmarks for each face
        landmarks = predictor(gray, face)
        
        # Convert the landmarks to numpy array
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append([x, y])
        points = np.array(points)
        
        # Perform Delaunay triangulation
        tri = Delaunay(points)
        
        # Plot the face landmarks and the triangulation
        plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()
        
    # Display the frame with landmarks
    cv2.imshow('Landmarks', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
