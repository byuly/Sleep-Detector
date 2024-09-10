
import numpy as np
import cv2

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def draw_angle(frame, point1, point2, point3, angle, color):
    cv2.putText(frame, str(int(angle)), 
                (point2[0] - 50, point2[1] + 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.line(frame, point1, point2, color, 2)
    cv2.line(frame, point2, point3, color, 2)

def show_start_screen():
    background = cv2.imread('data/posture.png') 
    start_screen = cv2.resize(background, (640, 480))
    
    # Add welcome text
    cv2.putText(start_screen, "Welcome to Posture Corrector!", (180, 200),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(start_screen, "Get into your desired posture, calibration will start when you press 's'!", 
                (0, 260), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the start screen
    while True:
        cv2.imshow('Start Screen', start_screen)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to start
            break

    cv2.destroyWindow('Start Screen')

    def calculate_eye_aspect_ratio(eye_landmarks):
        # Indices for the eye landmarks from the face mesh model (assume you are using MediaPipe Face Mesh)
        # You need to adjust these indices if they are different in your implementation
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))

        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_eye_closure(eye_landmarks):
        EAR_THRESHOLD = 0.2  # You may need to adjust this threshold
        ear = calculate_eye_aspect_ratio(eye_landmarks)
    
        # Return True if the EAR is below the threshold (indicating eye closure), otherwise False
        return ear < EAR_THRESHOLD