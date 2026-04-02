#!/usr/bin/env python3
"""
Keyboard Drone Controller + Live Detection Camera (Gazebo Only)
================================================================
Fly your drone manually with WASD keys while viewing the camera feed
with YOLO person detection, bounding boxes, and telemetry overlay.

Controls:
    W / S       = Forward / Backward
    A / D       = Strafe Left / Right
    Q / E       = Rotate Left / Right
    R / F       = Climb / Descend
    SPACE       = Stop all movement (hover)
    T           = Arm + Takeoff to 4m
    L           = Land
    ESC         = Quit

Usage:
    python keyboard_fly.py

Requires:
    pip install dronekit pynput ultralytics
"""

import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence

import time
import threading
import os
os.environ["MAVLINK20"] = "1"

from dronekit import connect
from pymavlink import mavutil
from pynput import keyboard
import cv2
import numpy as np
from ultralytics import YOLO
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image


# ── Config ──
CONNECTION_STRING = '127.0.0.1:14550'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TAKEOFF_ALT = 4.0
STICK_OFFSET = 150
CONFIDENCE_THRESHOLD = 0.3
AREA_CLOSE = 40000
AREA_MEDIUM = 15000
AREA_FAR = 3000


# ============================================================================
# CAMERA (Gazebo)
# ============================================================================

latest_frame = None

def on_camera_image(msg):
    global latest_frame
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

gz_node = Node()
CAMERA_TOPIC = "/front_camera"
gz_node.subscribe(Image, CAMERA_TOPIC, on_camera_image)
print(f"[✓] Camera: Gazebo ({CAMERA_TOPIC})")


# ============================================================================
# YOLO MODEL
# ============================================================================

yolo_model = YOLO("yolo26n.pt")
print("[✓] Detector: YOLO26n (person)")


# ============================================================================
# CONNECT TO SITL
# ============================================================================

print(f"[*] Connecting to SITL on {CONNECTION_STRING}...")
vehicle = connect(CONNECTION_STRING, wait_ready=False)
time.sleep(3)
print("[✓] Connected!\n")

motor_pwm = [0, 0, 0, 0]
pwm_min = 1000
pwm_max = 2000

try:
    pwm_min = vehicle.parameters.get('MOT_PWM_MIN', 1000)
    pwm_max = vehicle.parameters.get('MOT_PWM_MAX', 2000)
except:
    pass

@vehicle.on_message('SERVO_OUTPUT_RAW')
def on_servo(self, name, message):
    global motor_pwm
    motor_pwm = [message.servo1_raw, message.servo2_raw,
                 message.servo3_raw, message.servo4_raw]


# ============================================================================
# DETECTION
# ============================================================================

def detect_target(frame):
    results = yolo_model.predict(
        frame, conf=CONFIDENCE_THRESHOLD, classes=[0],
        verbose=False, imgsz=320
    )
    largest_area = 0
    best = None
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    best = {
                        "cx": (x1+x2)//2, "cy": (y1+y2)//2, "area": area,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": confidence, "label": "PERSON"
                    }
    return best


# ============================================================================
# HUD DRAWING
# ============================================================================

def get_proximity_label(area):
    if area >= AREA_CLOSE:
        return "CLOSE - LAND", (0, 255, 0)
    elif area >= AREA_MEDIUM:
        return "MEDIUM - APPROACH", (0, 255, 255)
    elif area >= AREA_FAR:
        return "FAR - DETECTED", (0, 165, 255)
    else:
        return "DISTANT", (128, 128, 128)


def draw_detection(frame, result):
    x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
    confidence = result["confidence"]
    area = result["area"]
    proximity_label, color = get_proximity_label(area)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    cv2.drawMarker(frame, (result["cx"], result["cy"]), color, cv2.MARKER_CROSS, 20, 2)

    cv2.putText(frame, proximity_label, (x1, y1 - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
    cv2.putText(frame, f"PERSON {confidence:.0%}", (x1, y1 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
    w, h = x2 - x1, y2 - y1
    cv2.putText(frame, f"Area: {area:,}px  ({w}x{h})", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_hud(frame, detected, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status = "PERSON DETECTED" if detected else "SCANNING"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, "MANUAL  SIM", (FRAME_WIDTH - 185, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    cv2.putText(frame, f"FPS:{fps:.0f}", (FRAME_WIDTH - 60, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.drawMarker(frame, (cx, cy), (50, 50, 50), cv2.MARKER_CROSS, 30, 1)


def draw_telemetry(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_HEIGHT - 56), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Battery
    if vehicle.battery.level is not None:
        bat_pct = vehicle.battery.level
        bat_volt = vehicle.battery.voltage
        bat_color = (0,255,0) if bat_pct > 40 else (0,255,255) if bat_pct > 20 else (0,0,255)
        bat_text = f"BATTERY: {bat_pct}%"
        if bat_volt: bat_text += f" ({bat_volt:.1f}V)"
    else:
        bat_color = (255, 255, 255)
        bat_text = "BATTERY: N/A"
    cv2.putText(frame, bat_text, (10, FRAME_HEIGHT - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, bat_color, 1)

    # Altitude + heading
    alt = vehicle.location.global_relative_frame.alt
    hdg = vehicle.heading
    cv2.putText(frame, f"ALT:{alt:.1f}m  HDG:{hdg:03d}", (FRAME_WIDTH - 200, FRAME_HEIGHT - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Motors
    if any(p > 0 for p in motor_pwm):
        pwm_range = pwm_max - pwm_min if pwm_max > pwm_min else 1
        for i, pwm in enumerate(motor_pwm):
            pct = max(0, min(100, (pwm - pwm_min) / pwm_range * 100))
            m_color = (0,255,0) if pct < 70 else (0,255,255) if pct < 90 else (0,0,255)
            cv2.putText(frame, f"M{i+1}:{pct:.0f}%", (10 + i*155, FRAME_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, m_color, 1)
    else:
        cv2.putText(frame, "MOTORS: N/A", (10, FRAME_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)


# ============================================================================
# KEYBOARD CONTROL
# ============================================================================

keys_pressed = set()
running = True
flying = False

def set_mode(mode_name):
    mode_mapping = {
        'STABILIZE': 0, 'ALT_HOLD': 2, 'GUIDED': 4,
        'LAND': 9, 'RTL': 6, 'LOITER': 5,
    }
    vehicle._master.mav.set_mode_send(
        vehicle._master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_mapping[mode_name]
    )
    time.sleep(1)

def arm_and_takeoff():
    global flying
    vehicle.channels.overrides = {}
    keys_pressed.clear()
    time.sleep(0.5)

    print("\n[*] GUIDED → Arming → Takeoff...")
    set_mode('GUIDED')
    vehicle.armed = True
    timeout = time.time() + 10
    while not vehicle.armed:
        if time.time() > timeout:
            print("[✗] Arming timed out!")
            return
        time.sleep(0.5)

    vehicle.simple_takeoff(TAKEOFF_ALT)
    while True:
        if vehicle.location.global_relative_frame.alt >= TAKEOFF_ALT * 0.9:
            break
        time.sleep(0.5)

    set_mode('LOITER')
    flying = True
    print("[✓] Airborne! WASD to fly.\n")

def land():
    global flying
    flying = False
    vehicle.channels.overrides = {}
    keys_pressed.clear()
    print("\n[*] Landing...")
    set_mode('LAND')

def on_press(key):
    global running
    try:
        k = key.char.lower()
        if k == 't':
            arm_and_takeoff()
        elif k == 'l':
            land()
        else:
            keys_pressed.add(k)
    except AttributeError:
        if key == keyboard.Key.space:
            keys_pressed.clear()
        elif key == keyboard.Key.esc:
            running = False

def on_release(key):
    try:
        keys_pressed.discard(key.char.lower())
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ============================================================================
# MAIN LOOP
# ============================================================================

print("╔══════════════════════════════════════════╗")
print("║    KEYBOARD CONTROLLER + DETECTION CAM   ║")
print("╠══════════════════════════════════════════╣")
print("║  W/S     = Forward / Backward            ║")
print("║  A/D     = Strafe Left / Right            ║")
print("║  Q/E     = Rotate Left / Right            ║")
print("║  R/F     = Climb / Descend                ║")
print("║  SPACE   = Stop all movement              ║")
print("║  T       = Arm + Takeoff                  ║")
print("║  L       = Land                           ║")
print("║  ESC     = Quit                           ║")
print("╚══════════════════════════════════════════╝")
print("\nPress T to arm and takeoff!\n")

fps = 0
frame_count = 0
fps_timer = cv2.getTickCount()

try:
    while running:
        # ── RC overrides (only when airborne) ──
        if flying:
            ch1, ch2, ch3, ch4 = 1500, 1500, 1500, 1500
            if 'w' in keys_pressed: ch2 = 1500 - STICK_OFFSET
            if 's' in keys_pressed: ch2 = 1500 + STICK_OFFSET
            if 'a' in keys_pressed: ch1 = 1500 - STICK_OFFSET
            if 'd' in keys_pressed: ch1 = 1500 + STICK_OFFSET
            if 'q' in keys_pressed: ch4 = 1500 - STICK_OFFSET
            if 'e' in keys_pressed: ch4 = 1500 + STICK_OFFSET
            if 'r' in keys_pressed: ch3 = 1500 + STICK_OFFSET
            if 'f' in keys_pressed: ch3 = 1500 - STICK_OFFSET
            vehicle.channels.overrides = {'1': ch1, '2': ch2, '3': ch3, '4': ch4}

        # ── Camera + Detection ──
        if latest_frame is not None:
            frame = latest_frame.copy()
            result = detect_target(frame)
            detected = result is not None

            if detected:
                draw_detection(frame, result)

            frame_count += 1
            elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = cv2.getTickCount()

            draw_hud(frame, detected, fps)
            draw_telemetry(frame)
            cv2.imshow("SAR Manual Flight", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.02)

except KeyboardInterrupt:
    pass

# ── Cleanup ──
print("\n[*] Releasing controls...")
vehicle.channels.overrides = {}
cv2.destroyAllWindows()
vehicle.close()
print("[✓] Done.")
