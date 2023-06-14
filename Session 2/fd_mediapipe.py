import cv2
import mediapipe as mp

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection

# Initialize the Face Detection module
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Read the input image
img = cv2.imread('eyeGaze.jpg')

# Convert the input image from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pass the RGB image to the Face Detection module to detect faces
results = face_detection.process(img_rgb)

# Loop over the detections and draw rectangles around the detected faces in the original image
if results.detections:
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, c = img.shape
        bbox_xmin = int(bbox.xmin * w)
        bbox_ymin = int(bbox.ymin * h)
        bbox_width = int(bbox.width * w)
        bbox_height = int(bbox.height * h)
        cv2.rectangle(img, (bbox_xmin, bbox_ymin), (bbox_xmin + bbox_width, bbox_ymin + bbox_height), (0, 255, 0), 2)

# Display the output image with the detected faces
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
