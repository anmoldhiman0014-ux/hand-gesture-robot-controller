"""
main.py - Hand Gesture Robot Controller
Real-time hand gesture recognition → Robot command dashboard

HOW TO USE:
  1. Run collect_data.py to gather gesture data (or use rule-based mode directly)
  2. Run train_model.py to train the SVM
  3. Run this file: python main.py
  
  WITHOUT training data → uses rule-based landmark detection (works immediately!)
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import os
import time
import math
import joblib
from collections import deque

# ─── Constants ──────────────────────────────────────────────────────────────
W, H = 1280, 720
CAM_W, CAM_H = 640, 480
DASH_X = CAM_W + 20

COMMANDS = {
    "STOP":     {"color": (220, 60, 60),   "icon": "■", "desc": "Fist closed"},
    "FORWARD":  {"color": (60, 220, 120),  "icon": "▲", "desc": "Palm open up"},
    "BACKWARD": {"color": (60, 150, 220),  "icon": "▼", "desc": "Palm open down"},
    "LEFT":     {"color": (220, 200, 60),  "icon": "◄", "desc": "Point left"},
    "RIGHT":    {"color": (220, 140, 60),  "icon": "►", "desc": "Point right"},
    "ROTATE":   {"color": (180, 60, 220),  "icon": "↻", "desc": "Two fingers up"},
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ─── Rule-Based Gesture Recognition (no training needed) ─────────────────────
def rule_based_gesture(landmarks):
    """Classify gesture using geometric rules on hand landmarks."""
    lm = landmarks.landmark

    def dist(a, b):
        return math.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)

    # Finger tip and knuckle indices
    tips = [4, 8, 12, 16, 20]
    knuckles = [3, 6, 10, 14, 18]
    mcp = [1, 5, 9, 13, 17]

    # Check which fingers are extended
    fingers_up = []
    # Thumb (special case - compare x coords)
    fingers_up.append(1 if lm[4].x < lm[3].x else 0)  # right hand
    # Other fingers
    for i in range(1, 5):
        fingers_up.append(1 if lm[tips[i]].y < lm[knuckles[i]].y else 0)

    total_up = sum(fingers_up)

    # Wrist y for orientation
    wrist_y = lm[0].y
    index_tip_y = lm[8].y
    pinky_tip_y = lm[20].y
    index_tip_x = lm[8].x
    pinky_tip_x = lm[20].x
    middle_tip_y = lm[12].y

    if total_up == 0:
        return "STOP"
    elif total_up >= 4:
        # All fingers up - direction based on tilt
        hand_tilt = lm[5].y - lm[17].y  # index MCP vs pinky MCP
        if abs(lm[8].x - lm[0].x) > 0.2:
            if lm[8].x < lm[0].x:
                return "LEFT"
            else:
                return "RIGHT"
        if middle_tip_y < wrist_y - 0.15:
            return "FORWARD"
        else:
            return "BACKWARD"
    elif fingers_up[1] == 1 and fingers_up[2] == 1 and total_up == 2:
        return "ROTATE"
    elif fingers_up[1] == 1 and total_up == 1:
        # Index only - direction
        dx = lm[8].x - lm[5].x
        dy = lm[8].y - lm[5].y
        if abs(dx) > abs(dy):
            return "LEFT" if dx < 0 else "RIGHT"
        return "FORWARD" if dy < 0 else "BACKWARD"
    elif total_up >= 3:
        if middle_tip_y < wrist_y:
            return "FORWARD"
        return "BACKWARD"

    return "STOP"


# ─── ML-Based Gesture Recognition ────────────────────────────────────────────
def load_ml_model():
    try:
        svm = joblib.load("models/gesture_svm.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return svm, scaler
    except:
        return None, None


GESTURE_NAMES_IDX = {0: "STOP", 1: "FORWARD", 2: "BACKWARD", 3: "LEFT", 4: "RIGHT", 5: "ROTATE"}

def ml_gesture(landmarks, model, scaler):
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    X = scaler.transform([features])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return GESTURE_NAMES_IDX[pred], prob


# ─── Pygame Dashboard ─────────────────────────────────────────────────────────
class RobotDashboard:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.current_cmd = "STOP"
        self.confidence = 0.0
        self.cmd_history = deque(maxlen=30)
        self.robot_angle = 0
        self.robot_x = DASH_X + 200
        self.robot_y = 350
        self.robot_vel = [0, 0]
        self.trail = deque(maxlen=60)
        self.frame_count = 0
        self.session_start = time.time()
        self.cmd_counts = {c: 0 for c in COMMANDS}

    def update(self, cmd, conf):
        self.frame_count += 1
        if cmd != self.current_cmd:
            self.cmd_history.append((cmd, time.time()))
            self.cmd_counts[cmd] += 1
        self.current_cmd = cmd
        self.confidence = conf

        # Move robot
        speed = 1.5
        if cmd == "FORWARD":
            self.robot_y -= speed
        elif cmd == "BACKWARD":
            self.robot_y += speed
        elif cmd == "LEFT":
            self.robot_x -= speed
        elif cmd == "RIGHT":
            self.robot_x += speed
        elif cmd == "ROTATE":
            self.robot_angle += 2

        # Bounds
        self.robot_x = max(DASH_X + 20, min(W - 20, self.robot_x))
        self.robot_y = max(200, min(H - 20, self.robot_y))

        self.trail.append((int(self.robot_x), int(self.robot_y)))

    def draw(self):
        # Background panel
        pygame.draw.rect(self.screen, (8, 12, 28), (DASH_X, 0, W - DASH_X, H))
        pygame.draw.line(self.screen, (30, 60, 100), (DASH_X, 0), (DASH_X, H), 2)

        cmd = self.current_cmd
        cmd_color = COMMANDS[cmd]["color"]

        # Title
        title = self.fonts["large"].render("🤖 ROBOT CONTROLLER", True, (0, 200, 255))
        self.screen.blit(title, (DASH_X + 10, 8))

        # Current command card
        card_rect = pygame.Rect(DASH_X + 10, 50, 290, 90)
        pygame.draw.rect(self.screen, (15, 20, 50), card_rect, border_radius=12)
        pygame.draw.rect(self.screen, cmd_color, card_rect, 2, border_radius=12)

        icon_surf = self.fonts["huge"].render(COMMANDS[cmd]["icon"], True, cmd_color)
        self.screen.blit(icon_surf, (DASH_X + 20, 58))

        cmd_surf = self.fonts["medium"].render(cmd, True, cmd_color)
        self.screen.blit(cmd_surf, (DASH_X + 80, 62))

        desc_surf = self.fonts["small"].render(COMMANDS[cmd]["desc"], True, (160, 160, 180))
        self.screen.blit(desc_surf, (DASH_X + 80, 95))

        # Confidence bar
        bar_y = 155
        self.screen.blit(self.fonts["tiny"].render("CONFIDENCE", True, (100, 120, 160)), (DASH_X + 10, bar_y))
        pygame.draw.rect(self.screen, (20, 25, 50), (DASH_X + 10, bar_y + 18, 290, 12), border_radius=6)
        conf_w = int(290 * self.confidence)
        if conf_w > 0:
            pygame.draw.rect(self.screen, cmd_color, (DASH_X + 10, bar_y + 18, conf_w, 12), border_radius=6)
        conf_text = self.fonts["tiny"].render(f"{self.confidence*100:.0f}%", True, (200, 200, 220))
        self.screen.blit(conf_text, (DASH_X + 305, bar_y + 16))

        # Robot arena
        arena_rect = pygame.Rect(DASH_X + 10, 195, 295, 250)
        pygame.draw.rect(self.screen, (10, 15, 35), arena_rect, border_radius=8)
        pygame.draw.rect(self.screen, (30, 50, 90), arena_rect, 1, border_radius=8)

        # Grid lines
        for gx in range(DASH_X + 10, DASH_X + 305, 30):
            pygame.draw.line(self.screen, (15, 22, 45), (gx, 195), (gx, 445))
        for gy in range(195, 445, 30):
            pygame.draw.line(self.screen, (15, 22, 45), (DASH_X + 10, gy), (DASH_X + 305, gy))

        # Trail
        pts = list(self.trail)
        if len(pts) > 2:
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = tuple(int(c * alpha) for c in cmd_color)
                pygame.draw.line(self.screen, color, pts[i - 1], pts[i], 2)

        # Robot body (clipped to arena)
        rx, ry = int(self.robot_x), int(self.robot_y)
        rx = max(arena_rect.left + 15, min(arena_rect.right - 15, rx))
        ry = max(arena_rect.top + 15, min(arena_rect.bottom - 15, ry))

        angle_rad = math.radians(self.robot_angle)
        # Draw robot as triangle
        pts_robot = []
        for da in [0, 140, 220]:
            a = math.radians(self.robot_angle + da)
            pts_robot.append((rx + 14 * math.sin(a), ry - 14 * math.cos(a)))
        pygame.draw.polygon(self.screen, cmd_color, pts_robot)
        pygame.draw.polygon(self.screen, (255, 255, 255), pts_robot, 1)

        # Compass
        compass_x, compass_y = DASH_X + 275, 215
        for label, angle in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            ca = math.radians(angle)
            lx = compass_x + 20 * math.sin(ca)
            ly = compass_y - 20 * math.cos(ca)
            self.screen.blit(self.fonts["tiny"].render(label, True, (80, 100, 140)), (lx - 5, ly - 7))
        pygame.draw.circle(self.screen, (30, 50, 80), (compass_x, compass_y), 18, 1)
        needle = math.radians(self.robot_angle)
        nx = compass_x + 14 * math.sin(needle)
        ny = compass_y - 14 * math.cos(needle)
        pygame.draw.line(self.screen, (0, 220, 255), (compass_x, compass_y), (int(nx), int(ny)), 2)
        pygame.draw.circle(self.screen, (0, 220, 255), (compass_x, compass_y), 3)

        # Stats
        elapsed = int(time.time() - self.session_start)
        m, s = elapsed // 60, elapsed % 60
        stats_y = 460
        self.screen.blit(self.fonts["tiny"].render("SESSION STATS", True, (100, 120, 160)), (DASH_X + 10, stats_y))
        stats_y += 18
        stats = [
            ("Time", f"{m:02d}:{s:02d}"),
            ("Frames", str(self.frame_count)),
            ("FPS", f"~30"),
        ]
        for label, val in stats:
            self.screen.blit(self.fonts["tiny"].render(label + ":", True, (120, 140, 170)), (DASH_X + 10, stats_y))
            self.screen.blit(self.fonts["tiny"].render(val, True, (200, 220, 255)), (DASH_X + 110, stats_y))
            stats_y += 16

        # Command frequency bars
        stats_y += 8
        self.screen.blit(self.fonts["tiny"].render("COMMAND USAGE", True, (100, 120, 160)), (DASH_X + 10, stats_y))
        stats_y += 18
        max_count = max(self.cmd_counts.values()) or 1
        for cmd_name, count in self.cmd_counts.items():
            bar_w = int((count / max_count) * 140)
            color = COMMANDS[cmd_name]["color"]
            self.screen.blit(self.fonts["tiny"].render(cmd_name[:3], True, color), (DASH_X + 10, stats_y))
            pygame.draw.rect(self.screen, (20, 25, 50), (DASH_X + 50, stats_y + 2, 140, 10), border_radius=3)
            if bar_w > 0:
                pygame.draw.rect(self.screen, color, (DASH_X + 50, stats_y + 2, bar_w, 10), border_radius=3)
            self.screen.blit(self.fonts["tiny"].render(str(count), True, (150, 160, 180)), (DASH_X + 200, stats_y))
            stats_y += 16

        # Mode indicator
        mode_text = "ML MODEL" if hasattr(self, 'using_ml') and self.using_ml else "RULE-BASED"
        mode_color = (0, 220, 120) if mode_text == "ML MODEL" else (220, 180, 0)
        mode_surf = self.fonts["tiny"].render(f"MODE: {mode_text}", True, mode_color)
        self.screen.blit(mode_surf, (DASH_X + 10, H - 25))
        pygame.draw.circle(self.screen, mode_color, (DASH_X + 10 + mode_surf.get_width() + 8, H - 18), 4)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Hand Gesture Robot Controller | IIT Mandi")
    clock = pygame.time.Clock()

    fonts = {
        "huge":   pygame.font.SysFont("segoeui", 40, bold=True),
        "large":  pygame.font.SysFont("segoeui", 22, bold=True),
        "medium": pygame.font.SysFont("segoeui", 26, bold=True),
        "small":  pygame.font.SysFont("segoeui", 16),
        "tiny":   pygame.font.SysFont("segoeui", 13),
    }

    # Try to load ML model
    svm_model, scaler = load_ml_model()
    use_ml = svm_model is not None
    print("✅ ML model loaded!" if use_ml else "⚠️  No model found — using rule-based recognition")
    print("   Run collect_data.py + train_model.py to enable ML mode\n")

    dashboard = RobotDashboard(screen, fonts)
    dashboard.using_ml = use_ml

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75,
                           min_tracking_confidence=0.6, model_complexity=1)

    # Smoothing buffer
    cmd_buffer = deque(maxlen=5)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_gesture = "STOP"
        confidence = 0.85

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                # Custom drawing
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=5),
                    mp_draw.DrawingSpec(color=(255, 255, 100), thickness=2))

                if use_ml:
                    gesture, conf = ml_gesture(hl, svm_model, scaler)
                    current_gesture = gesture
                    confidence = conf
                else:
                    current_gesture = rule_based_gesture(hl)
                    confidence = 1.0

        cmd_buffer.append(current_gesture)
        smoothed = max(set(cmd_buffer), key=list(cmd_buffer).count)
        dashboard.update(smoothed, confidence)

        # Convert frame to Pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        frame_surface = pygame.transform.scale(frame_surface, (CAM_W, CAM_H))

        screen.fill((6, 8, 20))

        # Camera feed with border
        cmd_color = COMMANDS[smoothed]["color"]
        pygame.draw.rect(screen, cmd_color, (-2, -2, CAM_W + 4, CAM_H + 4), 3)
        screen.blit(frame_surface, (0, 0))

        # Gesture overlay on camera
        overlay = pygame.Surface((CAM_W, 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, CAM_H - 60))
        icon = COMMANDS[smoothed]["icon"]
        gest_text = fonts["medium"].render(f"{icon}  {smoothed}", True, cmd_color)
        screen.blit(gest_text, (15, CAM_H - 45))

        # Instruction panel
        inst_y = CAM_H + 10
        screen.blit(fonts["small"].render("GESTURE GUIDE", True, (60, 100, 160)), (10, inst_y))
        inst_y += 22
        guides = [
            ("■ STOP",     "Close fist"),
            ("▲ FORWARD",  "Open palm up"),
            ("▼ BACKWARD", "Open palm down"),
            ("◄ LEFT",     "Point left"),
            ("► RIGHT",    "Point right"),
            ("↻ ROTATE",   "2 fingers up"),
        ]
        for g, d in guides:
            color = COMMANDS[g.split()[-1]]["color"]
            screen.blit(fonts["tiny"].render(g, True, color), (10, inst_y))
            screen.blit(fonts["tiny"].render(d, True, (130, 140, 160)), (130, inst_y))
            inst_y += 15

        # Footer
        screen.blit(fonts["tiny"].render("ESC to quit  |  Hand Gesture Robot Controller  |  Built with MediaPipe + SVM",
                                          True, (50, 60, 90)), (10, H - 20))

        dashboard.draw()
        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
