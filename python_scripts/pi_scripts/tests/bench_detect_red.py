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
import cv2
import numpy as np
from picamera2 import Picamera2
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '/dev/ttyAMA0'  # RPi5 via TELEM2
BAUD_RATE = 921600                  # RPi5 via TELEM2

# Bench test parameters
THROTTLE_PERCENT = 5        # Low throttle for bench test (0-100)
SPIN_DURATION = 20          # Seconds to keep motors spinning
ORIGINAL_ARMING_CHECK = None

# Video output
VIDEO_FILENAME = 'bench_test_recording.mp4'

# ============================================================================
# ── FUNCTIONS ──
# ============================================================================

def set_mode(vehicle, mode_name):
    """Change the drone's flight mode by sending a raw MAVLink command."""
    mode_mapping = {'GUIDED_NOGPS': 20, 'GUIDED': 4, 'LAND': 9, 'RTL': 6}
    vehicle._master.mav.set_mode_send(
        vehicle._master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_mapping[mode_name]
    )

def detect_red(frame):
    """
    Detect a red object in the frame.
    Draws bounding box on the frame if found.
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

# ============================================================================
# ── MAIN ──
# ============================================================================

def main():
    global ORIGINAL_ARMING_CHECK

    # ── CONNECT TO PIXHAWK ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING} at {BAUD_RATE} baud...")
    vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True)
    print("[✓] Connected!\n")

    # ── START CAMERA ──
    print("[*] Starting camera...")
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={'size': (640, 480), 'format': 'RGB888'}))
    cam.start()
    time.sleep(1)
    print("[✓] Camera ready.\n")

    # ── START VIDEO RECORDER ──
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 15.0, (640, 480))
    print(f"[*] Recording to {VIDEO_FILENAME}\n")

    # ── DISABLE ARMING CHECKS ──
    ORIGINAL_ARMING_CHECK = vehicle.parameters['ARMING_CHECK']
    print(f"[*] Original ARMING_CHECK value: {ORIGINAL_ARMING_CHECK}")
    print("[*] Disabling arming checks...")
    vehicle.parameters['ARMING_CHECK'] = 0
    time.sleep(2)
    print("[✓] Arming checks disabled.\n")

    # ── SET MODE TO STABILIZE ──
    print("[*] Setting mode to STABILIZE...")
    vehicle.mode = VehicleMode("STABILIZE")
    while vehicle.mode.name != "STABILIZE":
        time.sleep(0.5)
    print("[✓] Mode: STABILIZE\n")

    # ── ARM ──
    print("[*] Arming motors...")
    vehicle.armed = True
    timeout = 10
    start = time.time()
    while not vehicle.armed:
        if time.time() - start > timeout:
            print("[✗] Arming timed out!")
            cleanup(vehicle, cam, video_out)
            return
        time.sleep(0.5)
    print("[✓] Motors ARMED!\n")

    # ══════════════════════════════════════════════════════════════════════
    # SPIN MOTORS + CAMERA FEED + DETECTION + RECORD
    # ══════════════════════════════════════════════════════════════════════

    throttle_pwm = 1000 + int((THROTTLE_PERCENT / 100) * 1000)
    print(f"[*] Throttle: {THROTTLE_PERCENT}% (PWM: {throttle_pwm})")
    print(f"[*] Running for {SPIN_DURATION} seconds...\n")

    vehicle.channels.overrides['3'] = throttle_pwm

    spin_start = time.time()
    frame_count = 0

    while time.time() - spin_start < SPIN_DURATION:
        # Grab frame
        frame = cam.capture_array()
        elapsed = time.time() - spin_start

        # Run red detection (draws bounding box on frame if found)
        result = detect_red(frame)

        if result:
            print(f"  [{elapsed:.1f}s] 🔴 DETECTED! "
                  f"Pixel: ({result['cx']}, {result['cy']})  Area: {result['area']}")
        
        # Add recording overlay
        cv2.putText(frame, f"REC {elapsed:.1f}s / {SPIN_DURATION}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Record and display
        video_out.write(frame)
        cv2.imshow("Drone Camera", frame)
        cv2.waitKey(1)
        frame_count += 1

        # Print telemetry every 5 seconds
        if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.05):
            print(f"  [{elapsed:.1f}s] Battery: {vehicle.battery.voltage}V / "
                  f"{vehicle.battery.current}A  Frames: {frame_count}")

    print(f"\n[✓] Recording complete. {frame_count} frames captured.\n")

    # ── STOP MOTORS ──
    print("[*] Releasing throttle override...")
    vehicle.channels.overrides['3'] = None
    vehicle.channels.overrides = {}
    time.sleep(1)

    # ── DISARM ──
    print("[*] Disarming motors...")
    vehicle.armed = False
    while vehicle.armed:
        time.sleep(0.5)
    print("[✓] Motors DISARMED.\n")

    # ── CLEANUP ──
    cleanup(vehicle, cam, video_out)

def cleanup(vehicle, cam=None, video_out=None):
    """Restore arming checks, stop camera, save video, close connection."""
    global ORIGINAL_ARMING_CHECK

    cv2.destroyAllWindows()

    if video_out is not None:
        video_out.release()
        print(f"[✓] Video saved to {VIDEO_FILENAME}")

    if cam is not None:
        cam.stop()
        print("[✓] Camera stopped.")

    if ORIGINAL_ARMING_CHECK is not None:
        print(f"[*] Restoring ARMING_CHECK to {ORIGINAL_ARMING_CHECK}...")
        vehicle.parameters['ARMING_CHECK'] = ORIGINAL_ARMING_CHECK
        time.sleep(1)
        print("[✓] Arming checks restored.")

    print("[*] Closing connection...")
    vehicle.close()
    print("[✓] Done. Bench test complete.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        raise
