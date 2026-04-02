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
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ============================================================================
# ── CONFIG ──
# ============================================================================

CONNECTION_STRING = '/dev/ttyAMA0'  # RPi5 via TELEM2
BAUD_RATE = 921600                  # RPi5 via TELEM2
# CONNECTION_STRING = 'COM5'             # Windows via USB
# BAUD_RATE = 115200                     # Windows via USB

# BENCH TEST PARAMETERS
THROTTLE_PERCENT = 5        # Low throttle for bench test (0-100)
SPIN_DURATION = 15            # Seconds to keep motors spinning
ORIGINAL_ARMING_CHECK = None # Will store original value to restore later

def set_mode(vehicle, mode_name):
    """Change the drone's flight mode by sending a raw MAVLink command.
    
    Flight modes control what the drone does:
      - GUIDED: Accepts velocity/position commands from our script
      - LAND:   Descends and lands autonomously
      - RTL:    Returns to the takeoff point and lands (Return To Launch)
    
    We use raw MAVLink instead of DroneKit's vehicle.mode because
    DroneKit's mode setter can be unreliable/slow.
    """
    # Map human-readable mode names to ArduCopter's internal mode numbers
    mode_mapping = {'GUIDED_NOGPS' : 20, 'GUIDED': 4, 'LAND': 9, 'RTL': 6}
    
    # Send the mode change command directly via MAVLink
    vehicle._master.mav.set_mode_send(
        vehicle._master.target_system,                       # Target system ID (the drone)
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,   # Flag: we're setting a custom mode
        mode_mapping[mode_name]                               # The mode number to set
    )

def main():
    global ORIGINAL_ARMING_CHECK

    # ── CONNECT ──
    print(f"[*] Connecting to Pixhawk on {CONNECTION_STRING} at {BAUD_RATE} baud...")
    vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True)
    print("[✓] Connected!\n")

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
            print("[✗] Arming timed out! Check Pixhawk messages.")
            cleanup(vehicle)
            return
        time.sleep(0.5)
    print("[✓] Motors ARMED!\n")

    # ── OVERRIDE THROTTLE TO SPIN MOTORS ──
    # Channel 3 = throttle. Value 1000 = min, 2000 = max.
    throttle_pwm = 1000 + int((THROTTLE_PERCENT / 100) * 1000)
    print(f"[*] Setting throttle to {THROTTLE_PERCENT}% (PWM: {throttle_pwm})")
    print(f"[*] Motors will spin for {SPIN_DURATION} seconds...\n")

    vehicle.channels.overrides['3'] = throttle_pwm

    # Print telemetry during spin
    for i in range(SPIN_DURATION):
        print(f"  ── Second {i+1}/{SPIN_DURATION} ──")
        print(f"    Armed: {vehicle.armed}")
        print(f"    Mode:  {vehicle.mode.name}")
        print(f"    Battery: {vehicle.battery.voltage}V / {vehicle.battery.current}A")
        att = vehicle.attitude
        print(f"    Attitude: roll={att.roll:.2f} pitch={att.pitch:.2f} yaw={att.yaw:.2f}")
        time.sleep(1)

    # ── STOP MOTORS ──
    print("\n[*] Releasing throttle override...")
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
    cleanup(vehicle)

def cleanup(vehicle):
    """Restore arming checks and close connection."""
    global ORIGINAL_ARMING_CHECK
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
