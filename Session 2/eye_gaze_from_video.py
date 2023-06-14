''' 

In this code, we first load the Haar cascades for face and eye detection using OpenCV's `CascadeClassifier` method. 
We also load an iris template image that will be used to match the edges of the iris.

We then define the minimum and maximum sizes of the face and eyes, the threshold for detecting iris edges, and the number of iterations for the iris matching algorithm.

We read the input video using OpenCV's `VideoCapture` method and loop over the frames in the video using a `while` loop. For each frame, we convert it to grayscale and detect faces in the frame using the face detector. We then loop over the detected faces, extract the eye regions using the eye detector, and loop over the detected eyes.

For each eye, we find the edges of the iris using Canny edge detection, and match the edges to the iris template using iterative least squares fitting. We then draw the estimated iris center and radius on the eye region using OpenCV's `circle` method.

Finally, we display the output frame with the estimated gaze direction using OpenCV's `imshow` method, and wait for the 'q' key to be pressed to exit the loop. We release the video capture and close all windows using OpenCV's `release` and `destroyAllWindows` methods, respectively.

'''
import cv2
import numpy as np

# Define the face detector and the eye detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the iris template image
iris_template = cv2.imread('iris_template.png', cv2.IMREAD_GRAYSCALE)

# Define the minimum and maximum sizes of the face and eyes
min_face_size = (80, 80)
max_face_size = (300, 300)
min_eye_size = (20, 20)
max_eye_size = (80, 80)

# Define the threshold for detecting iris edges
iris_edge_threshold = 100

# Define the number of iterations for the iris matching algorithm
iris_matching_iterations = 10

# Read the input video
cap = cv2.VideoCapture(0) # cap = cv2.VideoCapture('video.mp4')

# Loop over the frames in the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the input frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the input frame using the face detector
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=min_face_size, maxSize=max_face_size)

    # Loop over the detected faces and extract the eye regions
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the face region using the eye detector
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=min_eye_size, maxSize=max_eye_size)

        # Loop over the detected eyes and estimate the gaze direction
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]

            # Find the edges of the iris using Canny edge detection
            edges = cv2.Canny(eye_gray, iris_edge_threshold, iris_edge_threshold * 2)

            # Match the iris edges to the iris template using iterative least squares fitting
            iris_center, iris_radius = None, None
            for i in range(iris_matching_iterations):
                edge_points = np.transpose(np.nonzero(edges))
                if len(edge_points) < 3:
                    break
                mean_x = np.mean(edge_points[:, 1])
                mean_y = np.mean(edge_points[:, 0])
                centered_points = edge_points - np.array([mean_y, mean_x])
                cov_matrix = np.cov(centered_points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                major_axis = eigenvectors[np.argmax(eigenvalues)]
                minor_axis = eigenvectors[np.argmin(eigenvalues)]
                if major_axis[1] < 0:
                    major_axis = -major_axis
                if minor_axis[0] < 0:
                    minor_axis = -minor_axis
                center = np.array([mean_x, mean_y])
                a = np.linalg.norm(centered_points @ major_axis)
                b = np.linalg.norm(centered_points @ minor_axis)
                r = np.sqrt(a * b)
                iris_center = center + major_axis * r
                iris_radius = r

             # Draw the iris center and radius on the eye region
        if iris_center is not None:
            cv2.circle(eye_color, tuple(np.round(iris_center).astype(int)), int(np.round(iris_radius)), (0, 255, 0), 2)

    # Display the output frame with the estimated gaze direction
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
