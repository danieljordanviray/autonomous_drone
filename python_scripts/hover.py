"""
Autonomous Flight Script - GPS GUIDED Mode
Takeoff → Hover → Land
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
import os
os.environ["MAVLINK20"] = "1"
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '127.0.0.1:14550'  # ArduPilot
# BAUD_RATE = 921600                  # RPi5 via TELEM2
# CONNECTION_STRING = 'COM5'             # Windows via USB
# BAUD_RATE = 115200                     # Windows via USB

TARGET_ALTITUDE = 1.5    # meters — higher than before since GPS holds position
HOVER_DURATION = 10      # seconds
TAKEOFF_TIMEOUT = 30     # seconds — abort if takeoff takes too long

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
    
    GPS fix types:
      0 = no GPS
      1 = no fix
      2 = 2D fix (not good enough for flight)
      3 = 3D fix (minimum for flight)
    
    We also wait for a minimum number of satellites for reliability.
    More satellites = more accurate position.
    """
    print("[*] Waiting for GPS lock...")
    while True:
        fix = vehicle.gps_0.fix_type
        sats = vehicle.gps_0.satellites_visible
        print(f"  GPS Fix: {fix} | Satellites: {sats}")

        # Need 3D fix and enough satellites
        if fix >= 3 and sats >= min_satellites:
            print(f"[✓] GPS locked: {fix}D fix, {sats} satellites\n")
            return True

        time.sleep(1)


def wait_for_ekf(vehicle):
    """
    Wait for the EKF (Extended Kalman Filter) to be healthy.
    
    The EKF fuses GPS, accelerometer, gyro, compass, and barometer
    into a single reliable position/attitude estimate. The FC won't
    arm in GUIDED mode unless the EKF is happy.
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
    
    Uses the NED (North-East-Down) coordinate frame:
      vx: velocity North (m/s)  — positive = north
      vy: velocity East (m/s)   — positive = east
      vz: velocity Down (m/s)   — positive = DOWN (so negative = climb)
    
    The FC handles all the low-level thrust/attitude to achieve this velocity.
    Send (0, 0, 0) to hold position (hover in place).
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,                                          # time_boot_ms
        vehicle._master.target_system,              # target system
        vehicle._master.target_component,           # target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,        # frame
        0b0000111111000111,                         # type_mask: velocity only
        0, 0, 0,                                    # position (ignored)
        vx, vy, vz,                                 # velocity
        0, 0, 0,                                    # acceleration (ignored)
        0, 0                                        # yaw, yaw_rate (ignored)
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
# ── MAIN ──
# ============================================================================

def main():

    # ── CONNECT ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING}")
    # vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True)
    vehicle = connect(CONNECTION_STRING, wait_ready=False) # SITL
    time.sleep(5)
    print("[✓] Connected!\n")

    # ── INITIAL TELEMETRY ──
    print_telemetry(vehicle)

    # ── WAIT FOR GPS AND EKF ──
    # These are the two things that MUST be ready before GUIDED mode works.
    # In GUIDED_NOGPS we skipped these — that's why everything was harder.
    wait_for_gps(vehicle)
    wait_for_ekf(vehicle)

    # ── SET MODE TO GUIDED ──
    # In GUIDED mode, the FC accepts position and velocity commands
    # and handles all stabilization, altitude hold, and position hold.
    if not set_mode(vehicle, 'GUIDED'):
        print("[✗] Could not enter GUIDED mode. Aborting.")
        vehicle.close()
        return

    # ── ARM ──
    # With GPS and EKF healthy, arming checks should pass naturally.
    # No need to disable ARMING_CHECK.
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

    # simple_takeoff() tells the FC "climb to this altitude."
    # The FC handles all motor control — thrust, stabilization, everything.
    # This is why GUIDED mode is so much easier than GUIDED_NOGPS.
    print(f"[*] Taking off to {TARGET_ALTITUDE}m...")
    vehicle.simple_takeoff(TARGET_ALTITUDE)

    takeoff_start = time.time()

    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Climbing... Alt: {altitude:.2f}m / {TARGET_ALTITUDE}m")

        if altitude >= TARGET_ALTITUDE * 0.95:
            print(f"[✓] Reached target altitude: {altitude:.2f}m\n")
            break

        # Safety timeout — if takeoff stalls, abort
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

    # In GUIDED mode, the FC holds position and altitude automatically.
    # We send zero velocity to explicitly say "stay here."
    # Even without sending anything, it would hold — but being explicit is safer.
    print(f"[*] Hovering for {HOVER_DURATION} seconds...")
    hover_start = time.time()

    while time.time() - hover_start < HOVER_DURATION:
        altitude = vehicle.location.global_relative_frame.alt
        elapsed = time.time() - hover_start
        print(f"  Hovering... Alt: {altitude:.2f}m  ({elapsed:.1f}s / {HOVER_DURATION}s)")

        # Zero velocity = hold position
        send_velocity(vehicle, 0, 0, 0)
        time.sleep(0.5)

    print(f"[✓] Hover complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    # LAND
    # ══════════════════════════════════════════════════════════════════════

    # LAND mode works properly with GPS — the FC descends, detects touchdown,
    # and auto-disarms. No manual thrust management needed.
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
