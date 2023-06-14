import mediapipe as mp
import cv2

# Load the MediaPipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Load the input image
image = cv2.imread("BT.jpg")

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect face landmarks
results = face_mesh.process(image_rgb)

# Iterate over the detected faces
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw the face mesh
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            idx1 = connection[0]
            idx2 = connection[1]
            x1 = int(face_landmarks.landmark[idx1].x * image.shape[1])
            y1 = int(face_landmarks.landmark[idx1].y * image.shape[0])
            x2 = int(face_landmarks.landmark[idx2].x * image.shape[1])
            y2 = int(face_landmarks.landmark[idx2].y * image.shape[0])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

