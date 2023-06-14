import cv2
import pygame
import numpy as np
from pygame.locals import *
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Create the Pygame window
window = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Hand Tracking')

# Initialize the hand tracking module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        # Read the frame from the camera
        ret, frame = camera.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # Draw landmarks on the hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                image_h, image_w, _ = frame.shape
                landmarks = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = min(int(landmark.x * image_w), 640)
                    y = min(int(landmark.y * image_h), 480)
                    landmarks.append((idx, x, y))

                    # Print the tracking point identifier on the screen
                    font = pygame.font.Font(None, 24)
                    text = font.render(str(idx), True, (0, 255, 0))
                    window.blit(text, (x, y))

                # Draw landmarks on the frame
                for _, x, y in landmarks:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Rotate the frame by 90 degrees counterclockwise
        frame = np.rot90(frame)

        # Convert the frame back to BGR for Pygame display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame in the Pygame window
        frame_pygame = pygame.surfarray.make_surface(frame_bgr)
        window.blit(frame_pygame, (0, 0))
        pygame.display.flip()

        # Exit the loop if 'q' is pressed
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_q:
                    camera.release()
                    pygame.quit()
                    exit()
