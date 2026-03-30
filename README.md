# 🤖 Hand Gesture Robot Controller

> **Real-time hand gesture recognition mapped to robot commands using MediaPipe + SVM + OpenCV**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-SVM-red?style=flat-square)

---

## 📌 About

This project demonstrates a **computer vision pipeline** that:
1. Detects **21 hand landmarks** in real-time using Google's MediaPipe
2. Classifies hand gestures using a trained **SVM classifier** (or rule-based fallback)
3. Maps gestures to **robot commands** and visualizes them on an animated Pygame dashboard

This simulates the control interface for a real ground robot — directly applicable to teleoperation research in robotics labs.

---

## 🎮 Gesture Commands

| Gesture | Command | Description |
|---------|---------|-------------|
| ✊ Closed Fist | **STOP** | All fingers closed |
| 🖐 Open Palm Up | **FORWARD** | All fingers extended upward |
| 🖐 Open Palm Down | **BACKWARD** | Hand facing downward |
| 👈 Point Left | **LEFT** | Index finger pointing left |
| 👉 Point Right | **RIGHT** | Index finger pointing right |
| ✌️ Two Fingers Up | **ROTATE** | Index + middle extended |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run immediately (Rule-Based Mode — no training needed!)
```bash
python main.py
```

### 3. (Optional) Collect & Train your own ML model
```bash
# Step 1: Collect gesture data (~200 samples per gesture)
python collect_data.py

# Step 2: Train SVM classifier
python train_model.py

# Step 3: Run with ML mode
python main.py
```

---

## 🧠 Technical Details

### Architecture
```
Webcam Feed → MediaPipe Hand Detection → 21 Landmark Extraction
     → Feature Vector (63-dim) → SVM Classifier (RBF kernel)
     → Gesture Label → Smoothing Buffer → Robot Command
     → Pygame Dashboard Visualization
```

### ML Model
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Features**: 63 normalized (x, y, z) coordinates of 21 hand landmarks
- **Training Accuracy**: ~97–99% with 200 samples per class
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Smoothing**: 5-frame majority vote to eliminate flicker

### Why SVM?
SVMs work exceptionally well for gesture recognition because:
- Hand landmark coordinates are **linearly separable in high-dimensional space** via RBF kernel
- Fast inference time (< 1ms) — ideal for real-time 30fps pipeline
- **Interpretable** and well-studied in robotics HCI literature

---

## 📁 Project Structure

```
hand-gesture-robot-controller/
├── main.py              # Live gesture → robot dashboard
├── collect_data.py      # Webcam-based gesture data collector
├── train_model.py       # SVM trainer + evaluation
├── requirements.txt
├── gesture_data.csv     # Generated after data collection
└── models/
    ├── gesture_svm.pkl  # Trained SVM model
    ├── scaler.pkl       # Feature normalizer
    └── confusion_matrix.png  # Model evaluation plot
```

---

## 🔬 Future Work
- [ ] Extend to 3D robot arm control (6-DOF)
- [ ] Replace SVM with LSTM for temporal gesture sequences
- [ ] Integrate with ROS (Robot Operating System) for real hardware
- [ ] Add bilateral teleoperation feedback

---

*Built as part of IIT Mandi Robotics & AI Internship Application*
