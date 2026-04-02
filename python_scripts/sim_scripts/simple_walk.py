"""
Autonomous Flight Script - GPS GUIDED Mode
Takeoff → Hover → Walk Forward 5m → Return to Launch → Land
Requires GPS lock.
"""

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
import math                                          # ADDED
import os
os.environ["MAVLINK20"] = "1"
from dronekit import connect, VehicleMode, LocationGlobalRelative  # CHANGED: added LocationGlobalRelative
from pymavlink import mavutil

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '127.0.0.1:14550'  # ArduPilot
# BAUD_RATE = 921600                  # RPi5 via TELEM2
# CONNECTION_STRING = 'COM5'             # Windows via USB
# BAUD_RATE = 115200                     # Windows via USB

TARGET_ALTITUDE = 1.5    # meters
HOVER_DURATION = 1      # seconds
TAKEOFF_TIMEOUT = 30     # seconds — abort if takeoff takes too long
WALK_DISTANCE = 5        # meters forward                    # ADDED
WALK_SPEED = 1.0         # m/s                               # ADDED

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
    """
    Wait until the GPS has a good fix before allowing flight.
    """
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
    """
    Wait for the EKF (Extended Kalman Filter) to be healthy.
    """
    print("[*] Waiting for EKF to converge...")
    while True:
        if vehicle.ekf_ok:
            print("[✓] EKF OK\n")
            return True
        print("  EKF not ready...")
        time.sleep(1)


def send_velocity(vehicle, vx, vy, vz):
    """
    Send a velocity command in GUIDED mode.
    Uses the NED (North-East-Down) coordinate frame.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        vehicle._master.target_system,
        vehicle._master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ── ADDED: three new functions for navigation ──

def get_distance_metres(loc1, loc2):
    """Approximate distance between two GPS locations in metres."""
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt(
        (dlat * 111320) ** 2 +
        (dlon * 111320 * math.cos(math.radians(loc1.lat))) ** 2
    )


def get_location_ahead(vehicle, distance_m):
    """
    Get a GPS coordinate 'distance_m' metres ahead of the drone's
    current heading.
    """
    current = vehicle.location.global_relative_frame
    heading_rad = math.radians(vehicle.heading)

    # heading 0 = North (+lat), 90 = East (+lon)
    dlat = distance_m * math.cos(heading_rad) / 111320.0
    dlon = distance_m * math.sin(heading_rad) / (111320.0 * math.cos(math.radians(current.lat)))

    target = LocationGlobalRelative(
        current.lat + dlat,
        current.lon + dlon,
        TARGET_ALTITUDE
    )
    return target


def goto_position(vehicle, target_location, speed=1.0):
    """Fly to a GPS position and wait until arrival."""
    print(f"[*] Flying to target at {speed} m/s...")
    vehicle.groundspeed = speed
    vehicle.simple_goto(target_location)

    while True:
        current = vehicle.location.global_relative_frame
        dist = get_distance_metres(current, target_location)
        alt = current.alt
        print(f"  Distance to target: {dist:.1f}m | Alt: {alt:.2f}m")

        if dist < 0.5:  # within 0.5m = arrived
            print("[✓] Reached target position\n")
            break
        time.sleep(0.5)

# ── END OF ADDED FUNCTIONS ──


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
# ── MAIN ──
# ============================================================================

def main():

    # ── CONNECT ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING}")
    vehicle = connect(CONNECTION_STRING, wait_ready=False) # SITL
    time.sleep(5)
    print("[✓] Connected!\n")

    # ── INITIAL TELEMETRY ──
    print_telemetry(vehicle)

    # ── WAIT FOR GPS AND EKF ──
    wait_for_gps(vehicle)
    wait_for_ekf(vehicle)

    # ── SAVE HOME POSITION ──                                    # ADDED
    home = vehicle.location.global_relative_frame                 # ADDED
    print(f"[*] Home saved: lat={home.lat:.6f} lon={home.lon:.6f}\n")  # ADDED

    # ── SET MODE TO GUIDED ──
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
            print("[✗] Arming timed out! Check pre-arm messages.")
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
            print("[✗] Takeoff timeout! Aborting — switching to LAND.")
            set_mode(vehicle, 'LAND')
            while vehicle.armed:
                time.sleep(0.5)
            vehicle.close()
            return

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
        time.sleep(0.5)

    print(f"[✓] Hover complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    # WALK FORWARD 5m                                              # ADDED
    # ══════════════════════════════════════════════════════════════════════

    print(f"[*] Walking forward {WALK_DISTANCE}m...")
    target = get_location_ahead(vehicle, WALK_DISTANCE)
    print(f"  Target: lat={target.lat:.6f} lon={target.lon:.6f} alt={target.alt}m")
    goto_position(vehicle, target, speed=WALK_SPEED)

    # Brief pause at forward position
    print("[*] Holding position for 3 seconds...")
    for i in range(6):
        send_velocity(vehicle, 0, 0, 0)
        time.sleep(0.5)
    print("[✓] Hold complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    # RETURN TO LAUNCH                                             # ADDED
    # ══════════════════════════════════════════════════════════════════════

    print("[*] Returning to launch position...")
    home_target = LocationGlobalRelative(home.lat, home.lon, TARGET_ALTITUDE)
    goto_position(vehicle, home_target, speed=WALK_SPEED)

    # Brief pause over home
    print("[*] Holding over home for 3 seconds...")
    for i in range(6):
        send_velocity(vehicle, 0, 0, 0)
        time.sleep(0.5)
    print("[✓] Back at home.\n")

    # ══════════════════════════════════════════════════════════════════════
    # LAND
    # ══════════════════════════════════════════════════════════════════════

    print("[*] Landing...")
    set_mode(vehicle, 'LAND')

    while vehicle.armed:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Descending... Alt: {altitude:.2f}m")
        time.sleep(0.5)

    print("[✓] Landed and disarmed.\n")

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
