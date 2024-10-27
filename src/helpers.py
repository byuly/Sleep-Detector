
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
    cv2.putText(start_screen, "Welcome to Sleep Detector!", (180, 200),
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
    
