import cv2

# Open the default camera (0) or a specific camera (1, 2, etc.)
cap = cv2.VideoCapture(0)

# Check if the camera was successfully opened
if not cap.isOpened():
    print('Error opening camera')

# Set the width and height of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read frames from the camera
while cap.isOpened():
    # Capture the frame-by-frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
