# 🚨 Sleep Detector - Driver Wellness Monitoring System 🚨

## Overview
This advanced computer vision application is designed to enhance driver safety by monitoring critical physiological and emotional states during long-haul drives. Initially started as a posture detector, the project has evolved into a comprehensive system that helps prevent fatigue-related accidents and provides insights into driver well-being.

---

## 🌟 Key Features

### 1. Drowsiness Detection
- **Real-time eye closure monitoring along with posture monitoring to increase accuracy **
- Tracks **shoulder and neck angles**
- **Immediate alerts** when prolonged eye closure is detected
- Provides **visual and audio alerts** for drowsiness detection
- Helps prevent **microsleep incidents**

### 2. Emotion Recognition
- Analyzes the driver's **emotional state in real-time**
- Tracks **emotional variations** during long drives
- Identifies potential **stress or fatigue indicators**

---

## 🤖 Technologies Used
- **Computer Vision**: OpenCV  
- **Machine Learning**:  
  - TensorFlow/Keras  
  - MediaPipe  
- **Programming Language**: Python  
- **Key Libraries**:  
  - `cv2`  
  - `mediapipe`  
  - `numpy`  
  - `tensorflow`

---

## 📊 Emotion Detection Spectrum
The system can recognize **7 distinct emotional states**:
- **Angry** 😠  
- **Disgusted** 🤢  
- **Fearful** 😨  
- **Happy** 😄  
- **Neutral** 😐  
- **Sad** 😔  
- **Surprised** 😲  

---

## 🔧 Installation

### Prerequisites
- **Python 3.8+**  
- **pip**  
- **Webcam**  

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/driver-wellness-monitor.git

# Install dependencies
pip install -r requirements.txt

# Run script
python src/main.py

