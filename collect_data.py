"""
collect_data.py - Gesture Dataset Collector
Run this script to collect training data for 6 hand gestures.
Press keys 0-5 to label gestures, 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

GESTURES = {
    0: "STOP",
    1: "FORWARD",
    2: "BACKWARD",
    3: "LEFT",
    4: "RIGHT",
    5: "ROTATE"
}

COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 100, 0),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (255, 0, 255)
}

DATA_FILE = "gesture_data.csv"
SAMPLES_PER_GESTURE = 200

def extract_landmarks(hand_landmarks):
    """Extract normalized landmark coordinates as feature vector."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def collect():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    current_label = -1
    counts = {i: 0 for i in range(6)}
    collecting = False

    # Create/load CSV
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f"x{i}" for i in range(63)] + ["label"]
            writer.writerow(header)

    print("\n🤖 GESTURE DATA COLLECTOR")
    print("=" * 40)
    for k, v in GESTURES.items():
        print(f"  Press [{k}] → Collect '{v}' gesture")
    print("  Press [SPACE] → Start/Stop collecting")
    print("  Press [q] → Quit")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))

                if collecting and current_label >= 0:
                    if counts[current_label] < SAMPLES_PER_GESTURE:
                        features = extract_landmarks(hl)
                        with open(DATA_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(features + [current_label])
                        counts[current_label] += 1

        # HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 200), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        label = current_label
        gesture_name = GESTURES.get(label, "NONE")
        color = COLORS.get(label, (200, 200, 200)) if label >= 0 else (200, 200, 200)

        status = "● RECORDING" if collecting else "○ PAUSED"
        scolor = (0, 255, 100) if collecting else (100, 100, 100)

        cv2.putText(frame, "GESTURE COLLECTOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)
        cv2.putText(frame, f"Mode: {gesture_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Count: {counts.get(label, 0)}/{SAMPLES_PER_GESTURE}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, status, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scolor, 2)

        y = 160
        for k, v in counts.items():
            bar_w = int((v / SAMPLES_PER_GESTURE) * 100)
            cv2.rectangle(frame, (10, y), (10 + bar_w, y + 6), COLORS[k], -1)
            cv2.putText(frame, f"{GESTURES[k][:3]}:{v}", (115, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y += 14

        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            collecting = not collecting
        elif key in [ord(str(i)) for i in range(6)]:
            current_label = int(chr(key))
            print(f"  → Now collecting: {GESTURES[current_label]}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Data saved to {DATA_FILE}")
    for k, v in counts.items():
        print(f"   {GESTURES[k]}: {v} samples")

if __name__ == "__main__":
    collect()
