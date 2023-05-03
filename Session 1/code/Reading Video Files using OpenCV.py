import cv2

# Open a video file
cap = cv2.VideoCapture('videos/street_traffic.mp4')

# Check if the video file was successfully opened
if not cap.isOpened():
    print('Error opening video file')

# Read frames from the video file
while cap.isOpened():
    # Capture the frame-by-frame from the video file
    ret, frame = cap.read()


    # Check if the frame was successfully captured
    if not ret:
        break
    
    image = cv2.putText(image, 'OpenCV', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Nous pouvons mettre de text sur la vidéo")

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()
