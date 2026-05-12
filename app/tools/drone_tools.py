import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "solar_farm")

# ── SWARM MESSAGE BUS ─────────────────────────────────────────────────────────
# Shared in-process dict. Drones write messages here, executor reads them.
# Format: { "drone_1": [ {type, zone, message}, ... ] }
# This is the agent-to-agent communication layer.
_swarm_inbox = {"drone_1": [], "drone_2": []}

def post_swarm_message(from_drone, to_drone, msg_type, zone, message):
    """Drone posts a message to another drone's inbox."""
    if to_drone in _swarm_inbox:
        _swarm_inbox[to_drone].append({
            "from": from_drone,
            "type": msg_type,   # "zone_done" | "battery_low" | "skip_zone"
            "zone": zone,
            "message": message
        })

def read_swarm_messages(drone_id):
    """Drain and return all messages for this drone."""
    msgs = _swarm_inbox.get(drone_id, [])
    _swarm_inbox[drone_id] = []
    return msgs

def reset_swarm():
    """Call at mission start."""
    _swarm_inbox["drone_1"] = []
    _swarm_inbox["drone_2"] = []


# ── DRONE ENV ─────────────────────────────────────────────────────────────────

BATTERY_PER_MOVE    = 8    # % per zone traversal
BATTERY_PER_CAPTURE = 5    # % per image capture
BATTERY_LOW_THRESH  = 25   # % — warn at this level
BATTERY_CRIT_THRESH = 15   # % — must return to dock

class DroneEnv:
    def __init__(self, drone_id="drone_1"):
        self.drone_id   = drone_id
        self.position   = "dock"
        self.images     = []
        self.battery    = 100.0        # percent
        self.zones_done = set()        # zones this drone has completed

    def move_to(self, zone):
        self.battery -= BATTERY_PER_MOVE
        self.battery = max(0, self.battery)
        self.position = zone

        # Warn partner if battery is low
        if self.battery <= BATTERY_LOW_THRESH:
            partner = "drone_2" if self.drone_id == "drone_1" else "drone_1"
            post_swarm_message(
                self.drone_id, partner,
                "battery_low", zone,
                f"{self.drone_id} battery at {self.battery:.0f}% near {zone}"
            )

        return f"Moved to {zone} [battery:{self.battery:.0f}%]"

    def capture_image(self, zone):
        self.battery -= BATTERY_PER_CAPTURE
        self.battery = max(0, self.battery)

        zone_path = os.path.join(DATA_DIR, zone)
        if os.path.exists(zone_path):
            files = [f for f in os.listdir(zone_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                img_path = os.path.join(zone_path, random.choice(files))
            else:
                img_path = f"mock_image_{zone}"
        else:
            img_path = f"mock_image_{zone}"

        self.images.append(img_path)
        return f"Captured image: {img_path} [battery:{self.battery:.0f}%]"

    def complete_zone(self, zone):
        """Mark zone done and notify partner so they don't re-inspect."""
        self.zones_done.add(zone)
        partner = "drone_2" if self.drone_id == "drone_1" else "drone_1"
        post_swarm_message(
            self.drone_id, partner,
            "zone_done", zone,
            f"{self.drone_id} completed {zone} — skip if assigned"
        )

    def needs_return(self):
        return self.battery <= BATTERY_CRIT_THRESH

    def return_to_dock(self):
        self.position = "dock"
        self.battery  = 100.0   # recharge at dock
        return "Returned to dock [recharged:100%]"