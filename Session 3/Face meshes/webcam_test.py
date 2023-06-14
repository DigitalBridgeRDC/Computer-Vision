import cv2
import mediapipe as mp

drawingModule = mp.solutions.drawing_utils
faceModule = mp.solutions.face_mesh

circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0, 255, 0))

# Initialize webcam
cap = cv2.VideoCapture(0)

with faceModule.FaceMesh(static_image_mode=False, max_num_faces=1) as face:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face.process(frame_rgb)

        # Draw the face mesh on the frame
        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                # Draw face landmarks
                drawingModule.draw_landmarks(frame, faceLandmarks, faceModule.FACEMESH_FACE_OVAL, circleDrawingSpec, lineDrawingSpec)
                drawingModule.draw_landmarks(frame, faceLandmarks, faceModule.FACEMESH_LIPS, circleDrawingSpec, lineDrawingSpec)
                drawingModule.draw_landmarks(frame, faceLandmarks, faceModule.FACEMESH_LEFT_EYE, circleDrawingSpec, lineDrawingSpec)
                drawingModule.draw_landmarks(frame, faceLandmarks, faceModule.FACEMESH_RIGHT_EYE, circleDrawingSpec, lineDrawingSpec)

        # Display the frame
        cv2.imshow('Face Mesh', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
