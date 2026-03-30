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
import os
os.environ["MAVLINK20"] = "1"
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# for camera / object detection
import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# ============================================================================
# ── CAMERA SUBSCRIBER ──
# ============================================================================

latest_frame = None

def on_camera_image(msg):
    """Callback — converts Gazebo camera frame to OpenCV format."""
    global latest_frame
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

gz_node = Node()
CAMERA_TOPIC = "/front_camera"
gz_node.subscribe(Image, CAMERA_TOPIC, on_camera_image)
print(f"[*] Subscribed to camera: {CAMERA_TOPIC}")

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '127.0.0.1:14550'  # ArduPilot SITL
TARGET_ALTITUDE = 1.5    # meters
HOVER_DURATION = 3       # seconds
TAKEOFF_TIMEOUT = 60     # seconds

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

    For 90° CW relative: condition_yaw(vehicle, 90, direction=1, relative=True)
    """
    msg = vehicle.message_factory.command_long_encode(
        0, 0,                                        # target system, component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,       # command
        0,                                            # confirmation
        heading,                                      # param1: degrees
        speed_deg_s,                                  # param2: deg/s (0 = default)
        direction,                                    # param3: 1=CW, -1=CCW
        1 if relative else 0,                         # param4: relative or absolute
        0, 0, 0                                       # params 5-7 unused
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
    """Return True if a red circular object is visible in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return False

    # Shape filter: ball's bounding box is roughly square
    # Line's bounding box is very wide or very tall
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = float(w) / h if h > 0 else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False

    # Fill ratio filter: at least 50% of the bounding box must be red pixels.
    # A circle fills ~78% of its bounding box. A diagonal line fills <10%.
    box_area = w * h
    red_in_box = cv2.countNonZero(mask[y:y+h, x:x+w])
    fill_ratio = red_in_box / box_area if box_area > 0 else 0
    if fill_ratio < 0.5:
        return False

    # Draw bounding box, center dot, and label on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    cv2.putText(frame, "RESCUE SUBJECT", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return True

def show_camera():
    """Display the latest camera frame in an OpenCV window."""
    if latest_frame is not None:
        cv2.imshow("Drone Camera", latest_frame)
        cv2.waitKey(1)

# ============================================================================
# ── MAIN ──
# ============================================================================

def main():

    # ── CONNECT ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING}")
    vehicle = connect(CONNECTION_STRING, wait_ready=False)
    time.sleep(5)
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
    # ROTATE
    # ══════════════════════════════════════════════════════════════════════

    start_heading = vehicle.heading
    rotate_by_deg = 300
    target_heading = (start_heading + rotate_by_deg) % 360

    print(f"[*] Rotating: {start_heading}° → {target_heading}°")
    condition_yaw(vehicle, heading=rotate_by_deg, direction=1, relative=True)

    # Wait for the rotation to complete by watching the heading
    yaw_start = time.time()
    while time.time() - yaw_start < 60:
        current = vehicle.heading
        altitude = vehicle.location.global_relative_frame.alt

        # ── DETECTION ──
        if latest_frame is not None:
            frame = latest_frame.copy()
            if detect_red(frame):
                print(f"  🔴 OBJECT DETECTED! Heading: {current}°")
            else:
                print(f"  Scanning... Heading: {current}°  No target.")
            # Show annotated frame (has bounding box if detected)
            cv2.imshow("Drone Camera", frame)
            cv2.waitKey(1)

        # Check if we're within of target heading
        diff = abs(current - target_heading)
        if diff > 180:
            diff = 360 - diff

        if diff <= 2:
            print(f"[✓] Rotation complete! Heading: {current}°\n")
            break

        time.sleep(0.5)

        # break out of the loop if we're 2 deg w/in target rotation
        # or 60s has passed
        # whichever comes first

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

    # ── CLOSE CAMERA ──
    cv2.destroyAllWindows()

    # ── CLEANUP ──
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
