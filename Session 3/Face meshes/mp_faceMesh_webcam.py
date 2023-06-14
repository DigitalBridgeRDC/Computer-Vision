import mediapipe as mp
import cv2

# Load the MediaPipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(frame_rgb)

    # Draw the face mesh on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                idx1 = connection[0]
                idx2 = connection[1]
                landmark1 = face_landmarks.landmark[idx1]
                landmark2 = face_landmarks.landmark[idx2]
                x1 = int(landmark1.x * frame.shape[1])
                y1 = int(landmark1.y * frame.shape[0])
                x2 = int(landmark2.x * frame.shape[1])
                y2 = int(landmark2.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with the face mesh
    cv2.imshow("Face Mesh", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
