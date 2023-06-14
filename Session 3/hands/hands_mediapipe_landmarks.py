import cv2
import mediapipe as mp

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
camera = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        # Read the frame from the camera
        ret, frame = camera.read()

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # Draw landmarks and print coordinates on the hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    image_h, image_w, _ = frame.shape
                    x = int(landmark.x * image_w)
                    y = int(landmark.y * image_h)
                    z = landmark.z
                    hand = "Right" if landmark.x < 0.5 else "Left"
                    coordinates = f"{hand} Hand - Landmark {idx}: X={x}, Y={y}, Z={z}"
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # cv2.putText(frame, coordinates, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    print(coordinates)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the windows
camera.release()
cv2.destroyAllWindows()
