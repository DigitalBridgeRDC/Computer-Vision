import cv2

# Open a video file
cap = cv2.VideoCapture('videos/street_traffic.mp4')

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (150, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2
   
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
    
    # Using cv2.putText() method
    frame = cv2.putText(frame, 'On peut meme mettre du texte sur la video', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()
