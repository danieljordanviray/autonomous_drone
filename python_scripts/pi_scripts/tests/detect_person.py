"""
Human Body Detection using YOLO26-nano on Raspberry Pi Camera
Detects people using a pre-trained neural network and displays
bounding box with area (useful for estimating distance/proximity).

Setup (run once on the Pi):
    pip install ultralytics --break-system-packages

First run will auto-download the yolo26n.pt model (requires internet once).

Usage:
    python detect_person.py
    Press 'q' to quit.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Optional DroneKit connection (for battery + motor telemetry)
# Falls back gracefully when not connected (bench testing)
# ──────────────────────────────────────────────
vehicle = None
motor_pwm = [0, 0, 0, 0]
pwm_min = 1000   # Fallback defaults
pwm_max = 2000

try:
    from dronekit import connect
    # Change this to your connection string:
    #   Real drone: '/dev/ttyAMA0' (Pi UART to Pixhawk)
    #   SITL sim:   'tcp:127.0.0.1:5762'
    #   Set to None to skip DroneKit entirely
    DRONEKIT_CONNECTION = '/dev/ttyAMA0'

    if DRONEKIT_CONNECTION:
        print(f"Connecting to Pixhawk on {DRONEKIT_CONNECTION}...")
        vehicle = connect(DRONEKIT_CONNECTION, wait_ready=True, baud=57600)
        print("Connected to Pixhawk.")

        # Read actual PWM range from Pixhawk parameters
        pwm_min = vehicle.parameters.get('MOT_PWM_MIN', 1000)
        pwm_max = vehicle.parameters.get('MOT_PWM_MAX', 2000)
        print(f"Motor PWM range: {pwm_min} - {pwm_max}")

        # Listen for motor outputs (SERVO_OUTPUT_RAW)
        @vehicle.on_message('SERVO_OUTPUT_RAW')
        def on_servo(self, name, message):
            global motor_pwm
            motor_pwm = [message.servo1_raw, message.servo2_raw,
                         message.servo3_raw, message.servo4_raw]
except ImportError:
    print("DroneKit not installed — running camera-only mode (no telemetry).")
except Exception as e:
    print(f"DroneKit connection failed: {e} — running camera-only mode.")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence to count as a detection
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PERSON_CLASS_ID = 0           # COCO class 0 = "person"

# Landing proximity thresholds (bounding box area in pixels)
# Tune these based on your actual flight altitude and camera FOV
AREA_CLOSE = 40000            # Person is close enough to land
AREA_MEDIUM = 15000           # Person detected, approaching
AREA_FAR = 3000               # Person detected, far away

# ──────────────────────────────────────────────
# Initialize camera
# ──────────────────────────────────────────────
cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={'size': (FRAME_WIDTH, FRAME_HEIGHT), 'format': 'RGB888'}
))
cam.start()

# ──────────────────────────────────────────────
# Load YOLO26-nano model
# ──────────────────────────────────────────────
# yolo26n = nano (fastest, edge-optimized, best for Pi)
# 43% faster CPU inference than YOLO11n, NMS-free, better small-object detection
# First run auto-downloads the model — requires internet once, then runs offline
model = YOLO("yolo26n.pt")

print("Model loaded. Starting detection...")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print("Press 'q' to quit.\n")

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def get_proximity_label(area):
    """Estimate proximity based on bounding box area."""
    if area >= AREA_CLOSE:
        return "CLOSE - LAND", (0, 255, 0)        # Green
    elif area >= AREA_MEDIUM:
        return "MEDIUM - APPROACH", (0, 255, 255)  # Yellow
    elif area >= AREA_FAR:
        return "FAR - DETECTED", (0, 165, 255)     # Orange
    else:
        return "DISTANT", (128, 128, 128)           # Gray


def draw_detection(frame, x1, y1, x2, y2, confidence, area):
    """Draw bounding box, labels, and proximity indicator on frame."""
    proximity_label, color = get_proximity_label(area)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Semi-transparent fill inside bounding box for visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    # Center crosshair
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

    # Labels stacked above bounding box (top to bottom):
    # Line 1: Proximity label
    cv2.putText(frame, proximity_label, (x1, y1 - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # Line 2: Class + confidence
    label = f"PERSON {confidence:.0%}"
    cv2.putText(frame, label, (x1, y1 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # Line 3: Area + dimensions
    w = x2 - x1
    h = y2 - y1
    cv2.putText(frame, f"Area: {area:,}px  ({w}x{h})", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return cx, cy


def draw_hud(frame, detected, fps, largest_area=0):
    """Draw heads-up display with status info."""
    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Detection status
    if detected:
        status = "SUBJECT LOCKED"
        status_color = (0, 255, 0)
    else:
        status = "SCANNING..."
        status_color = (0, 0, 255)

    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Frame center crosshair (aim point)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.drawMarker(frame, (cx, cy), (50, 50, 50),
                   cv2.MARKER_CROSS, 30, 1)


def draw_telemetry(frame):
    """Draw battery and motor telemetry at bottom-left of frame."""
    # Semi-transparent background bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_HEIGHT - 56), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Battery ──
    if vehicle is not None and vehicle.battery.level is not None:
        bat_pct = vehicle.battery.level
        bat_volt = vehicle.battery.voltage
        # Color: green > 40%, yellow 20-40%, red < 20%
        if bat_pct > 40:
            bat_color = (0, 255, 0)
        elif bat_pct > 20:
            bat_color = (0, 255, 255)
        else:
            bat_color = (0, 0, 255)
        bat_text = f"BATTERY: {bat_pct}%"
        if bat_volt is not None:
            bat_text += f" ({bat_volt:.1f}V)"
    else:
        bat_color = (255, 255, 255)
        bat_text = "BATTERY: N/A"

    cv2.putText(frame, bat_text, (10, FRAME_HEIGHT - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, bat_color, 1)

    # ── Motors ──
    if vehicle is not None and any(p > 0 for p in motor_pwm):
        pwm_range = pwm_max - pwm_min if pwm_max > pwm_min else 1
        for i, pwm in enumerate(motor_pwm):
            pct = max(0, min(100, (pwm - pwm_min) / pwm_range * 100))
            # Color: green < 70%, yellow 70-90%, red > 90%
            if pct < 70:
                m_color = (0, 255, 0)
            elif pct < 90:
                m_color = (0, 255, 255)
            else:
                m_color = (0, 0, 255)
            x_offset = 10 + i * 155
            cv2.putText(frame, f"M{i+1}: {pct:.0f}%", (x_offset, FRAME_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, m_color, 1)
    else:
        cv2.putText(frame, "MOTORS: N/A", (10, FRAME_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)


# ──────────────────────────────────────────────
# Main detection loop
# ──────────────────────────────────────────────
fps = 0
frame_count = 0
fps_timer = cv2.getTickCount()

while True:
    frame = cam.capture_array()

    # Run YOLO inference
    # verbose=False suppresses per-frame console output
    # classes=[0] filters to only detect "person" class (faster)
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        imgsz=320  # Smaller inference size = faster on Pi (trade-off: less accuracy at distance)
    )

    # Process detections
    detected = False
    largest_area = 0
    largest_detection = None

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)

                # Track the largest detection (closest person)
                if area > largest_area:
                    largest_area = area
                    largest_detection = (x1, y1, x2, y2, confidence, area)

    # Draw the largest (closest) person detection
    if largest_detection:
        detected = True
        x1, y1, x2, y2, conf, area = largest_detection
        cx, cy = draw_detection(frame, x1, y1, x2, y2, conf, area)

        # Print to console (useful for debugging without display)
        print(f"Person detected: center=({cx},{cy}) area={area:,} "
              f"conf={conf:.0%} proximity={get_proximity_label(area)[0]}")

    # Calculate FPS
    frame_count += 1
    elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_timer = cv2.getTickCount()

    # Draw HUD
    draw_hud(frame, detected, fps, largest_area)
    draw_telemetry(frame)

    # Display
    cv2.imshow("Person Detection - YOLO26n", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
cam.stop()
cv2.destroyAllWindows()
if vehicle:
    vehicle.close()
print("Detection stopped.")
