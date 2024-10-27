import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import helpers
import time 
import threading
from keras.models import model_from_json
import tensorflow as tf

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


with open("model/emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Initializing mediapipe for pose, and setting up video capture
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initializing mediapipe for face
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

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
ALERT_COOLDOWN = 5  
sound_file = 'src/ding.mp3'  
text_display_duration = 10
start_time = None
blink_counter = 0
eye_closed = False
frames_eye_closed = 0

# Defining indices as constnats for the eye top and bottom landmarks
LEFT_EYE_TOP = 160
LEFT_EYE_BOTTOM = 144
RIGHT_EYE_TOP = 385
RIGHT_EYE_BOTTOM = 380
EYE_CLOSED_FRAMES_THRESHOLD = 30


def sound(file_path): 
    # Threading for async soundplay since the video lags when playing sound directly
    sound_thread = threading.Thread(target=playsound, args=(file_path,), daemon=True)
    sound_thread.start()

helpers.show_start_screen()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: #if frame capture fails continue to next ITERATION of while loop
        continue
    face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    results = pose.process(frame)
    results2 = face_mesh.process(frame)

    if results.pose_landmarks and results2.multi_face_landmarks:
        landmarks = results.pose_landmarks.landmark
        face_landmarks = results2.multi_face_landmarks[0].landmark

        # Extracting x,y landmarks of position, normalized -> actual coordinates
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))

        # Eye landmarks normalized -> actual coordinates
        left_eye_top = (int(face_landmarks[LEFT_EYE_TOP].x * frame.shape[1]),
                        int(face_landmarks[LEFT_EYE_TOP].y * frame.shape[0]))
        left_eye_bottom = (int(face_landmarks[LEFT_EYE_BOTTOM].x * frame.shape[1]),
                           int(face_landmarks[LEFT_EYE_BOTTOM].y * frame.shape[0]))
        right_eye_top = (int(face_landmarks[RIGHT_EYE_TOP].x * frame.shape[1]),
                         int(face_landmarks[RIGHT_EYE_TOP].y * frame.shape[0]))
        right_eye_bottom = (int(face_landmarks[RIGHT_EYE_BOTTOM].x * frame.shape[1]),
                            int(face_landmarks[RIGHT_EYE_BOTTOM].y * frame.shape[0]))

        # Calculate the vertical eye aspect ratio
        left_eye_aspect_ratio = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
        right_eye_aspect_ratio = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))

        # Threshold value to detect closed eyes (adjust based on test observations)
        eye_aspect_ratio_threshold = 10 # fine-tune this if eye is detected too often / too less

        if left_eye_aspect_ratio < eye_aspect_ratio_threshold and right_eye_aspect_ratio < eye_aspect_ratio_threshold:
            frames_eye_closed += 1
        else:
            frames_eye_closed = 0

        # Check if eyes have been closed for the required number of frames
        eye_closed = frames_eye_closed >= EYE_CLOSED_FRAMES_THRESHOLD

        # Angle Calculation - in helpers python file
        shoulder_angle_left = helpers.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle_left = helpers.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        shoulder_angle_right = helpers.calculate_angle(right_shoulder, left_shoulder, (left_shoulder[0], 0))
        neck_angle_right = helpers.calculate_angle(right_ear, right_shoulder, (right_shoulder[0], 0))

        # Calibration process for 60 frames
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

        # Drawing landmarks onto video for testing
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        helpers.draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle_left, (255, 255, 255))
        helpers.draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle_left, (255, 255, 255))
        helpers.draw_angle(frame, right_shoulder, midpoint, (midpoint[0], 0), shoulder_angle_right, (255, 255, 255))
        helpers.draw_angle(frame, right_ear, right_shoulder, (right_shoulder[0], 0), neck_angle_right, (255, 255, 255))
        # Draw eye landmarks
        cv2.circle(frame, left_eye_top, 3, (0, 255, 0), -1)  # Green for left eye top
        cv2.circle(frame, left_eye_bottom, 3, (0, 0, 255), -1)  # Red for left eye bottom
        cv2.circle(frame, right_eye_top, 3, (0, 255, 0), -1)  # Green for right eye top
        cv2.circle(frame, right_eye_bottom, 3, (0, 0, 255), -1)  # Red for right eye bottom
        
        # Display closed-eye alert
        if eye_closed:
            cv2.putText(frame, "Eyes Closed for 30 frames!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        if is_calibrated:
            current_time = time.time()
            poor_posture_detected = False

            if shoulder_angle_left <= shoulder_threshold_left or neck_angle_left <= neck_threshold_left:
                poor_posture_detected = True
            if shoulder_angle_right <= shoulder_threshold_right or neck_angle_right <= neck_threshold_right:
                poor_posture_detected = True

            if poor_posture_detected and eye_closed:
                status = "Poor Posture!!"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > ALERT_COOLDOWN:  # 10-second cooldown, can be changed in declaration above
                    start_time = False
                    sound('data/ding.mp3')
                    last_alert_time = current_time
            elif poor_posture_detected: 
                status = "Poor Posture"
                color = (126, 0, 126)  # Green
                start_time = True
            else: 
                status = "Good Posture"
                color = (0, 255, 0)  # Green
                start_time = True
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA)

            # predict the emotions

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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

