#!/usr/bin/env python3
"""
Unified SAR (Search and Rescue) Mission Script
===============================================
One script for both simulation (Gazebo) and real-world (Raspberry Pi) flights.

Modes:
    --mode sim    → Gazebo camera + YOLO26n person detection + SITL connection
    --mode real   → Pi Camera + YOLO26n person detection + Pixhawk UART

Usage:
python sar_mission.py --mode sim                    # Gazebo + SITL
python sar_mission.py --mode real                   # Pi Camera + Pixhawk
python sar_mission.py --mode real --altitude 5      # Override hover altitude
python sar_mission.py --mode sim --no-record        # Skip video recording

Setup (run once):
    pip install ultralytics dronekit pymavlink --break-system-packages
"""

# ============================================================================
# IMPORTS
# ============================================================================

import collections
import collections.abc

# DroneKit compatibility fix for Python 3.10+
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence

import argparse
import time
import math
import threading
import os
os.environ["MAVLINK20"] = "1"

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="Unified SAR Mission Script")
parser.add_argument('--mode', choices=['sim', 'real'], required=True,
                    help="'sim' for Gazebo SITL, 'real' for Pi + Pixhawk")
parser.add_argument('--altitude', type=float, default=None,
                    help="Target hover altitude in meters (default: 4.0)")
parser.add_argument('--no-record', action='store_true',
                    help="Disable video recording")
args = parser.parse_args()

MODE = args.mode
print(f"\n{'='*60}")
print(f"  SAR MISSION — {'SIMULATION' if MODE == 'sim' else 'REAL FLIGHT'}")
print(f"{'='*60}\n")


# ============================================================================
# CONFIGURATION
# ============================================================================

# ── Mode-dependent settings ──
if MODE == 'sim':
    CONNECTION_STRING = '127.0.0.1:14550'
    BAUD_RATE = None
    CONNECT_WAIT_READY = False
else:
    CONNECTION_STRING = '/dev/ttyAMA0'
    BAUD_RATE = 921600
    CONNECT_WAIT_READY = True

# ── Shared settings ──
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_ALTITUDE = args.altitude or 4.0
HOVER_DURATION = 3          # seconds
TAKEOFF_TIMEOUT = 60        # seconds
CAMERA_HFOV_DEG = 120.0
CAMERA_HALF_FOV = CAMERA_HFOV_DEG / 2.0
APPROACH_SPEED = 0.5        # m/s
LOST_THRESHOLD = 10.0       # seconds before re-scan
LAND_AREA_THRESHOLD = 3000  # pixel area — target is "close enough" to land

# ── YOLO settings ──
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0          # COCO class 0 = "person"

# ── Proximity thresholds (HUD display) ──
AREA_CLOSE = 40000
AREA_MEDIUM = 15000
AREA_FAR = 3000


# ============================================================================
# CAMERA SETUP
# ============================================================================

latest_frame = None

if MODE == 'sim':
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image

    def on_camera_image(msg):
        global latest_frame
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gz_node = Node()
    CAMERA_TOPIC = "/front_camera"
    gz_node.subscribe(Image, CAMERA_TOPIC, on_camera_image)
    print(f"[✓] Camera: Gazebo ({CAMERA_TOPIC})")

else:
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={'size': (FRAME_WIDTH, FRAME_HEIGHT), 'format': 'RGB888'}
    ))
    cam.start()
    time.sleep(1)
    print("[✓] Camera: Pi Camera")

    def camera_loop():
        global latest_frame
        while True:
            latest_frame = cam.capture_array()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()


# ============================================================================
# VIDEO RECORDING
# ============================================================================

video_out = None
if not args.no_record:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    VIDEO_FILENAME = f'sar_flight_{MODE}_{timestamp}.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 15.0, (FRAME_WIDTH, FRAME_HEIGHT))
    print(f"[✓] Recording: {VIDEO_FILENAME}")
else:
    print("[*] Recording: disabled")


# ============================================================================
# YOLO MODEL
# ============================================================================

yolo_model = YOLO("yolo26n.pt")
print("[✓] Detector: YOLO26n (person)")


# ============================================================================
# TELEMETRY GLOBALS
# ============================================================================

motor_pwm = [0, 0, 0, 0]
pwm_min = 1000
pwm_max = 2000
mission_state = "INITIALIZING"


# ============================================================================
# DETECTION
# ============================================================================

def detect_target(frame):
    """
    Detect a person using YOLO26n neural network.
    Returns dict with cx, cy, area, x1, y1, x2, y2, confidence or None.
    """
    results = yolo_model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        imgsz=320
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
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    best = {
                        "cx": cx, "cy": cy, "area": area,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": confidence,
                        "label": "PERSON"
                    }
    return best


# ============================================================================
# HUD / OVERLAY FUNCTIONS
# ============================================================================

def get_proximity_label(area):
    """Estimate proximity based on bounding box area."""
    if area >= AREA_CLOSE:
        return "CLOSE - LAND", (0, 255, 0)
    elif area >= AREA_MEDIUM:
        return "MEDIUM - APPROACH", (0, 255, 255)
    elif area >= AREA_FAR:
        return "FAR - DETECTED", (0, 165, 255)
    else:
        return "DISTANT", (128, 128, 128)


def draw_detection(frame, result):
    """Draw bounding box, labels, and proximity indicator on frame."""
    x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
    confidence = result["confidence"]
    area = result["area"]
    label_text = result["label"]

    proximity_label, color = get_proximity_label(area)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    # Center crosshair
    cx, cy = result["cx"], result["cy"]
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

    # Labels stacked above bounding box
    cv2.putText(frame, proximity_label, (x1, y1 - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    cv2.putText(frame, f"{label_text} {confidence:.0%}", (x1, y1 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    w = x2 - x1
    h = y2 - y1
    cv2.putText(frame, f"Area: {area:,}px  ({w}x{h})", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_hud(frame, detected, fps):
    """Draw heads-up display with status info."""
    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Mission state
    state_color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, mission_state, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

    # Mode badge
    mode_text = "SIM" if MODE == 'sim' else "REAL"
    mode_color = (255, 200, 0) if MODE == 'sim' else (0, 200, 255)
    cv2.putText(frame, mode_text, (FRAME_WIDTH - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Frame center crosshair
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.drawMarker(frame, (cx, cy), (50, 50, 50), cv2.MARKER_CROSS, 30, 1)


def draw_telemetry(frame, vehicle):
    """Draw battery and motor telemetry at bottom of frame."""
    # Semi-transparent background bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_HEIGHT - 56), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Battery
    if vehicle is not None and vehicle.battery.level is not None:
        bat_pct = vehicle.battery.level
        bat_volt = vehicle.battery.voltage
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

    # Altitude
    if vehicle is not None:
        alt = vehicle.location.global_relative_frame.alt
        cv2.putText(frame, f"ALT: {alt:.1f}m", (FRAME_WIDTH - 120, FRAME_HEIGHT - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Motors
    if any(p > 0 for p in motor_pwm):
        pwm_range = pwm_max - pwm_min if pwm_max > pwm_min else 1
        for i, pwm in enumerate(motor_pwm):
            pct = max(0, min(100, (pwm - pwm_min) / pwm_range * 100))
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


def render_frame(frame, detected, fps, vehicle):
    """Apply all HUD overlays, record, and display the frame."""
    draw_hud(frame, detected, fps)
    draw_telemetry(frame, vehicle)
    if video_out is not None:
        video_out.write(frame)
    cv2.imshow("SAR Mission", frame)
    cv2.waitKey(1)


# ============================================================================
# FLIGHT FUNCTIONS
# ============================================================================

def set_mode(vehicle, mode_name):
    """Change flight mode via raw MAVLink command."""
    mode_mapping = {
        'STABILIZE': 0, 'ALT_HOLD': 2, 'GUIDED': 4,
        'LAND': 9, 'RTL': 6, 'GUIDED_NOGPS': 20,
    }
    vehicle._master.mav.set_mode_send(
        vehicle._master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_mapping[mode_name]
    )
    timeout = 5
    start = time.time()
    while vehicle.mode.name != mode_name:
        if time.time() - start > timeout:
            print(f"[!] Mode change to {mode_name} timed out")
            return False
        time.sleep(0.5)
    print(f"[✓] Mode: {mode_name}")
    return True


def wait_for_gps(vehicle, min_satellites=6):
    """Wait until the GPS has a 3D fix with enough satellites."""
    print("[*] Waiting for GPS lock...")
    while True:
        fix = vehicle.gps_0.fix_type
        sats = vehicle.gps_0.satellites_visible
        print(f"  GPS Fix: {fix} | Satellites: {sats}")
        if fix >= 3 and sats >= min_satellites:
            print(f"[✓] GPS locked: {fix}D fix, {sats} satellites\n")
            return True
        time.sleep(1)


def wait_for_ekf(vehicle):
    """Wait for the EKF to converge."""
    print("[*] Waiting for EKF to converge...")
    while True:
        if vehicle.ekf_ok:
            print("[✓] EKF OK\n")
            return True
        print("  EKF not ready...")
        time.sleep(1)


def send_velocity(vehicle, vx, vy, vz):
    """Send a NED velocity command. (0,0,0) = hold position."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, vehicle._master.target_system, vehicle._master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def condition_yaw(vehicle, heading, speed_deg_s=0, direction=1, relative=False):
    """Send a MAV_CMD_CONDITION_YAW command."""
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        heading, speed_deg_s, direction,
        1 if relative else 0,
        0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def print_telemetry(vehicle):
    """Print a snapshot of current drone state."""
    print("\n═══════════════ TELEMETRY ═══════════════")
    print(f"  Mode:            {vehicle.mode.name}")
    print(f"  Armed:           {vehicle.armed}")
    print(f"  GPS Fix:         {vehicle.gps_0.fix_type}")
    print(f"  Satellites:      {vehicle.gps_0.satellites_visible}")
    print(f"  EKF OK:          {vehicle.ekf_ok}")
    print(f"  Battery:         {vehicle.battery.voltage}V / {vehicle.battery.current}A")
    print(f"  Altitude:        {vehicle.location.global_relative_frame.alt:.2f}m")
    print(f"  Heading:         {vehicle.heading}°")
    att = vehicle.attitude
    print(f"  Attitude:        roll={att.roll:.2f} pitch={att.pitch:.2f} yaw={att.yaw:.2f}")
    print(f"  Groundspeed:     {vehicle.groundspeed:.1f} m/s")
    print(f"  System Status:   {vehicle.system_status.state}")
    print("═════════════════════════════════════════\n")


# ============================================================================
# SCAN FUNCTION
# ============================================================================

def scan_for_target(vehicle, fps):
    """
    Rotate 300° CW scanning for the target.
    Returns heading (int) where target was detected, or None.
    """
    global mission_state
    mission_state = "SCANNING"

    start_heading = vehicle.heading
    rotate_by_deg = 300
    target_heading = (start_heading + rotate_by_deg) % 360

    print(f"[*] Scanning: {start_heading}° → {target_heading}°")
    condition_yaw(vehicle, heading=rotate_by_deg, speed_deg_s=15, direction=1, relative=True)

    yaw_start = time.time()
    while time.time() - yaw_start < 60:
        current = vehicle.heading

        if latest_frame is not None:
            frame = latest_frame.copy()
            result = detect_target(frame)

            if result:
                mission_state = "TARGET DETECTED"
                draw_detection(frame, result)
                print(f"  🔴 TARGET DETECTED! Heading: {current}°")
                print(f"     Pixel: ({result['cx']}, {result['cy']})  Area: {result['area']}")
                render_frame(frame, True, fps, vehicle)
                return current
            else:
                print(f"  Scanning... Heading: {current}°  No target.")
                render_frame(frame, False, fps, vehicle)

        # Check if rotation is complete
        diff = abs(current - target_heading)
        if diff > 180:
            diff = 360 - diff
        if diff <= 2:
            print(f"[✓] Scan complete. Target not found.\n")
            return None

        time.sleep(0.5)

    print(f"[!] Scan timed out.\n")
    return None


# ============================================================================
# MAIN MISSION
# ============================================================================

def main():
    global mission_state, motor_pwm, pwm_min, pwm_max

    # ── CONNECT ──
    mission_state = "CONNECTING"
    print(f"[*] Connecting to {'SITL' if MODE == 'sim' else 'Pixhawk'} on {CONNECTION_STRING}...")

    if MODE == 'sim':
        vehicle = connect(CONNECTION_STRING, wait_ready=False)
        time.sleep(5)
    else:
        vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True)

    print("[✓] Connected!\n")

    # ── READ MOTOR PWM RANGE ──
    try:
        pwm_min = vehicle.parameters.get('MOT_PWM_MIN', 1000)
        pwm_max = vehicle.parameters.get('MOT_PWM_MAX', 2000)
        print(f"[✓] Motor PWM range: {pwm_min} - {pwm_max}")
    except Exception as e:
        print(f"[*] Could not read PWM params: {e} — using defaults.")

    # ── MOTOR TELEMETRY LISTENER ──
    @vehicle.on_message('SERVO_OUTPUT_RAW')
    def on_servo(self, name, message):
        global motor_pwm
        motor_pwm = [message.servo1_raw, message.servo2_raw,
                     message.servo3_raw, message.servo4_raw]

    # ── INITIAL TELEMETRY ──
    print_telemetry(vehicle)

    # ── PRE-FLIGHT CHECKS ──
    mission_state = "PRE-FLIGHT"
    wait_for_gps(vehicle)
    wait_for_ekf(vehicle)

    if not set_mode(vehicle, 'GUIDED'):
        print("[✗] Could not enter GUIDED mode. Aborting.")
        vehicle.close()
        return

    # ── ARM ──
    mission_state = "ARMING"
    print("[*] Arming motors...")
    vehicle.armed = True
    timeout = 10
    start = time.time()
    while not vehicle.armed:
        if time.time() - start > timeout:
            print("[✗] Arming timed out!")
            vehicle.close()
            return
        time.sleep(0.5)
    print("[✓] Motors ARMED!\n")

    # ── FPS TRACKING ──
    fps = 0
    frame_count = 0
    fps_timer = cv2.getTickCount()

    def update_fps():
        nonlocal fps, frame_count, fps_timer
        frame_count += 1
        elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = cv2.getTickCount()

    # ══════════════════════════════════════════════════════════════════════
    # TAKEOFF
    # ══════════════════════════════════════════════════════════════════════

    mission_state = "TAKING OFF"
    print(f"[*] Taking off to {TARGET_ALTITUDE}m...")
    vehicle.simple_takeoff(TARGET_ALTITUDE)

    takeoff_start = time.time()
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Climbing... Alt: {altitude:.2f}m / {TARGET_ALTITUDE}m")

        if altitude >= TARGET_ALTITUDE * 0.95:
            print(f"[✓] Reached target altitude: {altitude:.2f}m\n")
            break

        if time.time() - takeoff_start > TAKEOFF_TIMEOUT:
            print("[✗] Takeoff timeout! Landing.")
            set_mode(vehicle, 'LAND')
            while vehicle.armed:
                time.sleep(0.5)
            vehicle.close()
            return

        if latest_frame is not None:
            frame = latest_frame.copy()
            update_fps()
            render_frame(frame, False, fps, vehicle)

        time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════════
    # HOVER
    # ══════════════════════════════════════════════════════════════════════

    mission_state = "HOVERING"
    print(f"[*] Hovering for {HOVER_DURATION} seconds...")
    hover_start = time.time()
    while time.time() - hover_start < HOVER_DURATION:
        altitude = vehicle.location.global_relative_frame.alt
        elapsed = time.time() - hover_start
        print(f"  Hovering... Alt: {altitude:.2f}m  ({elapsed:.1f}s / {HOVER_DURATION}s)")
        send_velocity(vehicle, 0, 0, 0)

        if latest_frame is not None:
            frame = latest_frame.copy()
            update_fps()
            render_frame(frame, False, fps, vehicle)

        time.sleep(0.5)
    print(f"[✓] Hover complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    # SCAN → PURSUE → LAND
    # ══════════════════════════════════════════════════════════════════════

    detection_heading = scan_for_target(vehicle, fps)

    while detection_heading is not None:

        # Cancel any ongoing rotation
        condition_yaw(vehicle, vehicle.heading, relative=False)
        send_velocity(vehicle, 0, 0, 0)
        time.sleep(1)

        # ──────────────────────────────────────────────────────────────
        # PURSUE TARGET
        # ──────────────────────────────────────────────────────────────

        mission_state = "PURSUING"
        print(f"[*] Pursuing target at heading {detection_heading}°...")

        last_seen_area = 0
        lost_time = None

        while True:
            current_heading = vehicle.heading
            altitude = vehicle.location.global_relative_frame.alt

            if latest_frame is None:
                time.sleep(0.1)
                continue

            frame = latest_frame.copy()
            result = detect_target(frame)
            update_fps()

            if result:
                # ── Target visible: steer toward it ──
                lost_time = None
                last_seen_area = result["area"]
                draw_detection(frame, result)

                # Close enough — stop and land
                if last_seen_area > LAND_AREA_THRESHOLD:
                    mission_state = "TARGET REACHED"
                    print(f"  [*] Close to target (area: {last_seen_area}). Landing.")
                    send_velocity(vehicle, 0, 0, 0)
                    render_frame(frame, True, fps, vehicle)
                    detection_heading = None
                    break

                cx = result["cx"]
                frame_center = FRAME_WIDTH // 2
                offset = (cx - frame_center) / frame_center
                correction_deg = offset * CAMERA_HALF_FOV

                area_ratio = min(result["area"] / LAND_AREA_THRESHOLD, 1.0)
                speed = APPROACH_SPEED * (1.0 - 0.8 * area_ratio)

                bearing = current_heading + correction_deg
                bearing_rad = math.radians(bearing)
                vx = speed * math.cos(bearing_rad)
                vy = speed * math.sin(bearing_rad)

                send_velocity(vehicle, vx, vy, 0)
                render_frame(frame, True, fps, vehicle)

                print(f"  Pursuing... Heading: {current_heading}°  "
                      f"Offset: {offset:+.2f}  Correction: {correction_deg:+.1f}°  "
                      f"Speed: {speed:.2f} m/s  Area: {result['area']}  "
                      f"Alt: {altitude:.2f}m")

            else:
                # ── Target not visible ──
                if lost_time is None:
                    lost_time = time.time()

                elapsed_lost = time.time() - lost_time
                mission_state = "TARGET LOST"

                if last_seen_area > LAND_AREA_THRESHOLD:
                    mission_state = "TARGET BELOW"
                    print(f"  [*] Target below drone (last area: {last_seen_area}). Landing.")
                    send_velocity(vehicle, 0, 0, 0)
                    render_frame(frame, False, fps, vehicle)
                    detection_heading = None
                    break

                if elapsed_lost > LOST_THRESHOLD:
                    print(f"  [!] Lost target for {LOST_THRESHOLD}s. Re-scanning...")
                    send_velocity(vehicle, 0, 0, 0)
                    time.sleep(1)
                    render_frame(frame, False, fps, vehicle)
                    detection_heading = scan_for_target(vehicle, fps)
                    break

                render_frame(frame, False, fps, vehicle)
                print(f"  Searching... target lost ({elapsed_lost:.1f}s)  "
                      f"Alt: {altitude:.2f}m")

            time.sleep(0.2)

    # ══════════════════════════════════════════════════════════════════════
    # LAND
    # ══════════════════════════════════════════════════════════════════════

    mission_state = "LANDING"
    print("[*] Landing...")
    set_mode(vehicle, 'LAND')

    while vehicle.armed:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Descending... Alt: {altitude:.2f}m")

        if latest_frame is not None:
            frame = latest_frame.copy()
            update_fps()
            render_frame(frame, False, fps, vehicle)

        time.sleep(0.5)

    mission_state = "COMPLETE"
    print("[✓] Landed and disarmed.\n")

    # ── CLEANUP ──
    cv2.destroyAllWindows()
    if video_out is not None:
        video_out.release()
        print(f"[✓] Video saved to {VIDEO_FILENAME}")
    if MODE == 'real':
        cam.stop()

    print("[*] Closing connection...")
    vehicle.close()

    print(f"\n{'='*60}")
    print("MISSION COMPLETE")
    print(f"{'='*60}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        if video_out is not None:
            video_out.release()
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        if video_out is not None:
            video_out.release()
        raise
