'''

In this code, we first load the face detector and the landmark predictor using the `dlib` library. 
We also load the camera calibration matrix, which contains the intrinsic parameters of the camera.

We then define the 3D model points and the 2D image points for the eyes, nose, and screen. 
We use these points to estimate the 3D rotation and translation vectors using the `solvePnP` function from OpenCV, which implements the Perspective-n-Point algorithm.

We then project the 3D model points of the screen onto the image plane to determine the direction of gaze, and smooth the gaze direction using a simple low-pass filter.
Finally, we draw the estimated gaze direction on the frame using OpenCV's `line` function, and display the frame. We exit the loop when the 'q' key is pressed.

'''
import dlib
import cv2
import numpy as np

# Load the face detector and the landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the camera calibration matrix
camera_matrix = np.loadtxt('camera_matrix.txt')
dist_coeffs = np.loadtxt('dist_coeffs.txt')

# Define the 3D model points and the 2D image points for the eyes
eye_model_points = np.array([
    (-0.35, 0.15, 0),
    (0.35, 0.15, 0),
    (0, -0.2, 0)
], dtype=np.float32)
eye_image_points = np.array([
    (266, 198),
    (373, 197),
    (320, 299)
], dtype=np.float32)

# Define the 3D model points and the 2D image points for the nose
nose_model_points = np.array([
    (0, 0, 0),
    (0, 0, -20)
], dtype=np.float32)
nose_image_points = np.array([
    (320, 234),
    (320, 214)
], dtype=np.float32)

# Define the 3D model points and the 2D image points for the screen
screen_model_points = np.array([
    (-1, 1, 0),
    (1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0)
], dtype=np.float32)
screen_image_points = np.array([
    (0, 0),
    (640, 0),
    (640, 480),
    (0, 480)
], dtype=np.float32)

# Define the camera intrinsics matrix
image_size = (640, 480)
focal_length = image_size[1]
center = (image_size[1] / 2, image_size[0] / 2)
camera_intrinsics = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

# Initialize the 3D rotation and translation vectors
rvec = np.zeros((3,), dtype=np.float32)
tvec = np.zeros((3,), dtype=np.float32)

# Initialize the previous gaze direction
prev_gaze_direction = np.array([0, 0, 1], dtype=np.float32)

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Find faces in the frame
    faces = face_detector(frame, 0)

    # Loop over the detected faces
    for face in faces:
        # Find the facial landmarks
        landmarks = landmark_predictor(frame, face)

        # Extract the 2D image points for the eyes and the nose
        eye1_image_points = np.array([(landmarks.part(36).x, landmarks.part(36).y)], dtype=np.float32)
        eye2_image_points = np.array([(landmarks.part(45).x, landmarks.part(45).y)], dtype=np.float32)
        nose_image_points = np.array([(landmarks.part(30).x, landmarks.part(30).y)], dtype=np.float32)

        # Estimate the 3D rotation and translation vectors using the eyes and the nose
        _, rvec, tvec = cv2.solvePnP(np.vstack((eye_model_points, nose_model_points)), np.vstack((eye_image_points, nose_image_points)), camera_intrinsics, dist_coeffs, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)

    # Project the screen 3D model points onto the image plane
    screen_image_points_proj, _ = cv2.projectPoints(screen_model_points, rvec, tvec, camera_intrinsics, dist_coeffs)

    # Compute the 3D direction of gaze
    ray_direction = screen_image_points_proj[0][0] - np.array([center[0], center[1]])
    ray_direction = np.array([ray_direction[0], -ray_direction[1], -focal_length], dtype=np.float32)
    ray_direction /= np.linalg.norm(ray_direction)

    # Smooth the gaze direction using a simple low-pass filter
    gaze_direction = 0.7 * prev_gaze_direction + 0.3 * ray_direction
    prev_gaze_direction = gaze_direction

    # Draw the gaze direction on the frame
    x1, y1 = int(center[0]), int(center[1])
    x2, y2 = int(center[0] + gaze_direction[0] * 100), int(center[1] - gaze_direction[1] * 100)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
