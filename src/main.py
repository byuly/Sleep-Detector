import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import helpers
import time 
import threading

# Initializing mediapipe for pose, and setting up video capture
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0) # "0" opens default webcam

calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_frames = 0
is_calibrated = False
shoulder_threshold_left = 0
neck_threshold_left = 0
shoulder_threshold_right = 0
neck_threshold_right = 0
last_alert_time = 0
ALERT_COOLDOWN = 5  # 10 seconds cooldown between alerts
sound_file = 'src/ding.mp3'  # Make sure the sound file exists in working directory
text_display_duration = 10
start_time = None


def sound(file_path): 
    # Threading since the video lags when playing sound directly
    sound_thread = threading.Thread(target=playsound, args=(file_path,), daemon=True)
    sound_thread.start()

helpers.show_start_screen()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: #if frame capture fails continue to next ITERATION of while loop
        continue

    results = pose.process(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # extracting x,y landmarks of position, normalized -> actual coordinates
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))

        # Angle Calculation - in helpers python file 
        shoulder_angle_left = helpers.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle_left = helpers.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        shoulder_angle_right = helpers.calculate_angle(right_shoulder, left_shoulder, (left_shoulder[0], 0))
        neck_angle_right = helpers.calculate_angle(right_ear, right_shoulder, (right_shoulder[0], 0))

        # Calibration
        if not is_calibrated and calibration_frames < 60:
            calibration_shoulder_angles.append((shoulder_angle_left, shoulder_angle_right))
            calibration_neck_angles.append((neck_angle_left, neck_angle_right))
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/60", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold_left = np.mean([angle[0] for angle in calibration_shoulder_angles]) - 5
            neck_threshold_left = np.mean([angle[0] for angle in calibration_neck_angles]) - 5

            shoulder_threshold_right = np.mean([angle[1] for angle in calibration_shoulder_angles]) - 5
            neck_threshold_right = np.mean([angle[1] for angle in calibration_neck_angles]) - 5

            is_calibrated = True
            print(f"Calibration complete. Left Shoulder threshold: {shoulder_threshold_left:.1f}, Left Neck threshold: {neck_threshold_left:.1f}")
            print(f"Right Shoulder threshold: {shoulder_threshold_right:.1f}, Right Neck threshold: {neck_threshold_right:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        helpers.draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle_left, (255, 255, 255))
        helpers.draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle_left, (255, 255, 255))
        helpers.draw_angle(frame, right_shoulder, midpoint, (midpoint[0], 0), shoulder_angle_right, (255, 255, 255))
        helpers.draw_angle(frame, right_ear, right_shoulder, (right_shoulder[0], 0), neck_angle_right, (255, 255, 255))

        # Feedback, play sound when poor posture detected
        if is_calibrated:
            current_time = time.time()
            poor_posture_detected = False

            if shoulder_angle_left <= shoulder_threshold_left or neck_angle_left <= neck_threshold_left:
                poor_posture_detected = True
            if shoulder_angle_right <= shoulder_threshold_right or neck_angle_right <= neck_threshold_right:
                poor_posture_detected = True

            

            if poor_posture_detected:
                status = "Poor Posture!!"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > ALERT_COOLDOWN: # 10 second cooldown, can be changed in declaration above
                    start_time = False
                    sound('data/ding.mp3')
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green
                start_time = True

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA)
            if not start_time:
                cv2.putText(frame, "Poor posture detected! Please sit up straight.", (180, 255), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Left Shoulder Angle: {shoulder_angle_left:.1f}/{shoulder_threshold_left:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Left Neck Angle: {neck_angle_left:.1f}/{neck_threshold_left:.1f}", (10, 90), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Right Shoulder Angle: {shoulder_angle_right:.1f}/{shoulder_threshold_right:.1f}", (10, 120), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Right Neck Angle: {neck_angle_right:.1f}/{neck_threshold_right:.1f}", (10, 150), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

