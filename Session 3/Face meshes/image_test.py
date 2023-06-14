import cv2
import mediapipe as mp
 
drawingModule = mp.solutions.drawing_utils
faceModule = mp.solutions.face_mesh
 
circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0,255,0))
 
with faceModule.FaceMesh(static_image_mode=True) as face:
    image = cv2.imread("BT.jpg")
 
    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
    if results.multi_face_landmarks is not None:
        for faceLandmarks in results.multi_face_landmarks:
            # Draw face landmarks
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACEMESH_FACE_OVAL, circleDrawingSpec, lineDrawingSpec)
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACEMESH_LIPS, circleDrawingSpec, lineDrawingSpec)
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACEMESH_LEFT_EYE, circleDrawingSpec, lineDrawingSpec)
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACEMESH_RIGHT_EYE, circleDrawingSpec, lineDrawingSpec)
 
    cv2.imshow('Test image', image)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()



