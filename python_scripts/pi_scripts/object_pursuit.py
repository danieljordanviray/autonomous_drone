# ============================================================================
# ----- IMPORTS -----
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

import time
import math
import threading
import os
os.environ["MAVLINK20"] = "1"
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# Camera / object detection
import cv2
import numpy as np
from picamera2 import Picamera2

# ============================================================================
# ── CAMERA ──
# ============================================================================

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={'size': (640, 480), 'format': 'RGB888'}))
cam.start()
time.sleep(1)  # let camera warm up
print("[✓] Camera started.")

latest_frame = None

# Video recorder
VIDEO_FILENAME = 'flight_recording.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 15.0, (640, 480))
print(f"[*] Recording to {VIDEO_FILENAME}")

def camera_loop():
    """Background thread: continuously grab frames and record."""
    global latest_frame
    while True:
        frame = cam.capture_array()
        latest_frame = frame
        video_out.write(frame)

thread = threading.Thread(target=camera_loop, daemon=True)
thread.start()
print("[✓] Camera thread running.\n")

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '/dev/ttyAMA0'  # RPi5 via TELEM2
BAUD_RATE = 921600                  # RPi5 via TELEM2

TARGET_ALTITUDE = 1.5    # meters
HOVER_DURATION = 3       # seconds
TAKEOFF_TIMEOUT = 60     # seconds

# Camera: 120° horizontal FOV, 640x480
CAMERA_HFOV_DEG = 120.0
CAMERA_HALF_FOV = CAMERA_HFOV_DEG / 2.0  # 60°

# Pursuit
APPROACH_SPEED = 0.5      # m/s
LOST_THRESHOLD = 10.0     # seconds — re-scan if target lost this long
LAND_AREA_THRESHOLD = 2000  # pixel area — target is "close" when this big

# ============================================================================
# ── FUNCTIONS ──
# ============================================================================

def set_mode(vehicle, mode_name):
    """Change flight mode via raw MAVLink command."""
    mode_mapping = {
        'STABILIZE': 0,
        'ALT_HOLD': 2,
        'GUIDED': 4,
        'LAND': 9,
        'RTL': 6,
        'GUIDED_NOGPS': 20,
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
        0,
        vehicle._master.target_system,
        vehicle._master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,                         # type_mask: velocity only
        0, 0, 0,                                    # position (ignored)
        vx, vy, vz,                                 # velocity
        0, 0, 0,                                    # acceleration (ignored)
        0, 0                                        # yaw, yaw_rate (ignored)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def condition_yaw(vehicle, heading, speed_deg_s=0, direction=1, relative=False):
    """
    Send a MAV_CMD_CONDITION_YAW command.

    Straight from DroneKit docs:
        param1 = heading (degrees)
        param2 = yaw speed (deg/s, 0 = default speed)
        param3 = direction: 1 = CW, -1 = CCW
        param4 = 0 = absolute heading, 1 = relative to current
    """
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        heading,
        speed_deg_s,
        direction,
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


def detect_red(frame):
    """
    Detect a red object in the frame using real-world Pi camera thresholds.
    Draws bounding box on frame if found.
    Returns dict with cx, cy, area if found, or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Higher saturation (150) filters out skin tones
    mask1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:
        return None

    # Aspect ratio filter
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = float(w) / h if h > 0 else 0
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return None

    # Fill ratio — at least 30% of bounding box must be red
    box_area = w * h
    red_in_box = cv2.countNonZero(mask[y:y+h, x:x+w])
    fill_ratio = red_in_box / box_area if box_area > 0 else 0
    if fill_ratio < 0.3:
        return None

    # Draw bounding box, center dot, and label
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    cv2.putText(frame, f"RESCUE SUBJECT area:{int(area)}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return {"cx": cx, "cy": cy, "area": area}


def show_camera():
    """Display the latest camera frame in an OpenCV window."""
    if latest_frame is not None:
        cv2.imshow("Drone Camera", latest_frame)
        cv2.waitKey(1)


def scan_for_target(vehicle):
    """
    Rotate 300° CW scanning for the red target.
    Returns heading (int) where target was detected, or None if not found.
    """
    start_heading = vehicle.heading
    rotate_by_deg = 300
    target_heading = (start_heading + rotate_by_deg) % 360

    print(f"[*] Scanning: {start_heading}° → {target_heading}°")
    condition_yaw(vehicle, heading=rotate_by_deg, speed_deg_s=15, direction=1, relative=True)

    yaw_start = time.time()
    while time.time() - yaw_start < 60:
        current = vehicle.heading
        altitude = vehicle.location.global_relative_frame.alt

        if latest_frame is not None:
            frame = latest_frame.copy()
            result = detect_red(frame)
            if result:
                print(f"  🔴 RESCUE SUBJECT DETECTED! Heading: {current}°")
                print(f"     Pixel: ({result['cx']}, {result['cy']})  Area: {result['area']}")
                cv2.imshow("Drone Camera", frame)
                cv2.waitKey(1)
                return current
            else:
                print(f"  Scanning... Heading: {current}°  No target.")
            cv2.imshow("Drone Camera", frame)
            cv2.waitKey(1)

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
# ── MAIN ──
# ============================================================================

def main():

    # ── CONNECT ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING} at {BAUD_RATE} baud...")
    vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True)
    print("[✓] Connected!\n")

    # ── INITIAL TELEMETRY ──
    print_telemetry(vehicle)

    # ── PRE-FLIGHT CHECKS ──
    wait_for_gps(vehicle)
    wait_for_ekf(vehicle)

    if not set_mode(vehicle, 'GUIDED'):
        print("[✗] Could not enter GUIDED mode. Aborting.")
        vehicle.close()
        return

    # ── ARM ──
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

    # ══════════════════════════════════════════════════════════════════════
    # TAKEOFF
    # ══════════════════════════════════════════════════════════════════════

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

        show_camera()
        time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════════
    # HOVER
    # ══════════════════════════════════════════════════════════════════════

    print(f"[*] Hovering for {HOVER_DURATION} seconds...")
    hover_start = time.time()
    while time.time() - hover_start < HOVER_DURATION:
        altitude = vehicle.location.global_relative_frame.alt
        elapsed = time.time() - hover_start
        print(f"  Hovering... Alt: {altitude:.2f}m  ({elapsed:.1f}s / {HOVER_DURATION}s)")
        send_velocity(vehicle, 0, 0, 0)
        show_camera()
        time.sleep(0.5)
    print(f"[✓] Hover complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    # SCAN → PURSUE → LAND
    # ══════════════════════════════════════════════════════════════════════

    detection_heading = scan_for_target(vehicle)

    while detection_heading is not None:

        # Cancel any ongoing rotation by commanding current heading
        condition_yaw(vehicle, vehicle.heading, relative=False)
        send_velocity(vehicle, 0, 0, 0)
        time.sleep(1)

        # ──────────────────────────────────────────────────────────────
        # PURSUE RESCUE SUBJECT
        # ──────────────────────────────────────────────────────────────

        print(f"[*] Pursuing rescue subject at heading {detection_heading}°...")

        last_seen_area = 0
        lost_time = None

        while True:
            current_heading = vehicle.heading
            altitude = vehicle.location.global_relative_frame.alt

            if latest_frame is None:
                time.sleep(0.1)
                continue

            frame = latest_frame.copy()
            result = detect_red(frame)
            cv2.imshow("Drone Camera", frame)
            cv2.waitKey(1)

            if result:
                # ── Target visible: steer toward it ──
                lost_time = None
                last_seen_area = result["area"]

                # Close enough — stop and land while we can still see it
                if last_seen_area > LAND_AREA_THRESHOLD:
                    print(f"  [*] Close to target (area: {last_seen_area}). Landing.")
                    send_velocity(vehicle, 0, 0, 0)
                    detection_heading = None
                    break

                cx = result["cx"]
                frame_width = frame.shape[1]
                frame_center = frame_width // 2

                offset = (cx - frame_center) / frame_center
                correction_deg = offset * CAMERA_HALF_FOV

                # Proportional speed: slow down as we get closer
                area_ratio = min(result["area"] / LAND_AREA_THRESHOLD, 1.0)
                speed = APPROACH_SPEED * (1.0 - 0.8 * area_ratio)

                bearing = current_heading + correction_deg
                bearing_rad = math.radians(bearing)
                vx = speed * math.cos(bearing_rad)
                vy = speed * math.sin(bearing_rad)

                send_velocity(vehicle, vx, vy, 0)
                print(f"  Pursuing... Heading: {current_heading}°  "
                      f"Offset: {offset:+.2f}  Correction: {correction_deg:+.1f}°  "
                      f"Speed: {speed:.2f} m/s  Area: {result['area']}  "
                      f"Alt: {altitude:.2f}m")

            else:
                # ── Target not visible ──
                if lost_time is None:
                    lost_time = time.time()

                elapsed_lost = time.time() - lost_time

                if last_seen_area > LAND_AREA_THRESHOLD:
                    print(f"  [*] Target below drone (last area: {last_seen_area}). Landing.")
                    send_velocity(vehicle, 0, 0, 0)
                    detection_heading = None
                    break

                if elapsed_lost > LOST_THRESHOLD:
                    print(f"  [!] Lost target for {LOST_THRESHOLD}s. Re-scanning...")
                    send_velocity(vehicle, 0, 0, 0)
                    time.sleep(1)
                    detection_heading = scan_for_target(vehicle)
                    break

                print(f"  Searching... target lost ({elapsed_lost:.1f}s)  "
                      f"Alt: {altitude:.2f}m")

            time.sleep(0.2)

    # ══════════════════════════════════════════════════════════════════════
    # LAND
    # ══════════════════════════════════════════════════════════════════════

    print("[*] Landing...")
    set_mode(vehicle, 'LAND')

    while vehicle.armed:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Descending... Alt: {altitude:.2f}m")
        show_camera()
        time.sleep(0.5)

    print("[✓] Landed and disarmed.\n")

    # ── CLEANUP ──
    cv2.destroyAllWindows()
    video_out.release()
    cam.stop()
    print(f"[✓] Video saved to {VIDEO_FILENAME}")

    print("[*] Closing connection...")
    vehicle.close()

    print("\n" + "=" * 60)
    print("MISSION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        raise
