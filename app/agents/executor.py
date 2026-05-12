import os
import glob
from collections import Counter
from app.tools.drone_tools import DroneEnv, reset_swarm, read_swarm_messages, post_swarm_message
from app.agents.perception import perception_agent
from app.memory.panel_memory import update_panel
from app.memory.panel_graph_rag import refresh_panel_graph
from app.memory.zone_memory import update_zone
from app.observability.tracer import log_event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "solar_farm")

drones = {
    "drone_1": DroneEnv("drone_1"),
    "drone_2": DroneEnv("drone_2")
}


def get_panel_image(zone, panel_index):
    zone_path = os.path.join(DATA_DIR, zone)
    if not os.path.exists(zone_path):
        return f"mock_image_{zone}_{panel_index}"
    files = sorted(
        f for ext in ["*.jpg", "*.jpeg", "*.png"]
        for f in glob.glob(os.path.join(zone_path, ext))
    )
    if not files:
        return f"mock_image_{zone}_{panel_index}"
    return files[panel_index % len(files)]

from langsmith import traceable

@traceable(name="executor_agent", run_type="chain")
def executor_agent(state):
    plan             = state.get("plan", [])
    logs             = []
    analysis_results = []
    swarm_logs       = {}
    all_swarm_msgs   = []

    reset_swarm()
    for d in drones.values():
        d.battery    = 100.0
        d.zones_done = set()
        d.images     = []
        d.position   = "dock"

    zone_panel_buffer = {}   # zone -> list of panel results
    completed_zones   = set()  # zones fully done by any drone
    zone_scan_started = set()  # track which zones have emitted the scanning header

    i = 0
    while i < len(plan):
        task = plan[i]
        i += 1

        drone_id    = task.get("drone", "drone_1")
        action      = task.get("action")
        zone        = task.get("zone")
        panel_index = task.get("panel_index", 0)
        drone       = drones.get(drone_id)

        if drone is None:
            logs.append(f"[ERROR] invalid drone_id={drone_id}")
            continue

        #  Drain swarm inbox 
        msgs = read_swarm_messages(drone_id)
        for msg in msgs:
            all_swarm_msgs.append(msg)
            log_event("swarm", "message", msg)
            if msg["type"] == "battery_low":
                logs.append(f"[SWARM] {drone_id} received: {msg['message']}")

        #  Zone skip check: if zone completed by partner, skip block 
        if zone and zone in completed_zones and zone not in drones[drone_id].zones_done:
            # Find who completed it
            completer = None
            for msg in all_swarm_msgs:
                if msg.get("type") == "zone_done" and msg.get("zone") == zone \
                        and msg.get("from") != drone_id:
                    completer = msg.get("from")
                    break

            if completer:
                logs.append(
                    f"[SWARM] {drone_id} skipping {zone} "
                    f"— already done by {completer}"
                )
                post_swarm_message(
                    drone_id, completer, "skip_zone", zone,
                    f"{drone_id} skipping {zone} — confirmed done by {completer}"
                )
                all_swarm_msgs.append({
                    "from": drone_id, "to": completer,
                    "type": "skip_zone", "zone": zone,
                    "message": f"{drone_id} skipping {zone} — confirmed done by {completer}"
                })
                swarm_logs.setdefault(drone_id, []).append(f"skipped {zone}")
                # Skip all remaining tasks for this drone/zone combo
                while i < len(plan) and plan[i].get("zone") == zone \
                        and plan[i].get("drone") == drone_id:
                    i += 1
                continue

        #  MOVE 
        if action == "move_to":
            if drone.needs_return():
                logs.append(
                    f"[BATTERY CRITICAL] {drone_id} battery "
                    f"{drone.battery:.0f}% — forced return to dock"
                )
                drone.return_to_dock()
                logs.append(f"[DOCK] {drone_id} returned [recharged:100%]")
                swarm_logs.setdefault(drone_id, []).append("forced return")
                # Skip remaining tasks for this drone in this zone
                while i < len(plan) and plan[i].get("drone") == drone_id \
                        and plan[i].get("zone") == zone:
                    i += 1
                continue

            drone.move_to(zone)
            logs.append(f"[MOVE] {drone_id} -> {zone} [battery:{drone.battery:.0f}%]")
            swarm_logs.setdefault(drone_id, []).append(f"move {zone}")

        # CAPTURE PANEL 
        elif action == "capture_panel":
            # Emit scanning header once per zone per drone — on first panel
            scan_key = f"{drone_id}_{zone}"
            if scan_key not in zone_scan_started:
                logs.append(f"[SCAN_START] {drone_id} zone={zone}")
                zone_scan_started.add(scan_key)

            image_path = get_panel_image(zone, panel_index)
            drone.battery = max(0, drone.battery - 1.0)

            if "mock_image" not in image_path:
                analysis = perception_agent(image_path, zone=zone,
                                            panel_index=panel_index)
                if analysis["confidence"] < 0.55:
                    logs.append(
                        f"[LOW CONFIDENCE] retrying {zone} panel {panel_index}"
                    )
                    retry = perception_agent(image_path, zone=zone,
                                            panel_index=panel_index)
                    if retry["confidence"] > analysis["confidence"]:
                        analysis = retry
            else:
                analysis = {
                    "class": "Unknown", "confidence": 0.0,
                    "reasoning": "no image", "meta_note": ""
                }

            logs.append(
                f"[PANEL] {drone_id} zone={zone} panel={panel_index} "
                f"class={analysis.get('class','Unknown')} "
                f"confidence={analysis.get('confidence',0):.2f}"
            )

            # Write to panel memory
            update_panel(zone, panel_index, {
                "class":      analysis["class"],
                "confidence": analysis["confidence"],
                "meta_note":  analysis.get("meta_note", ""),
                "reasoning":  analysis.get("reasoning", ""),
                "image":      os.path.basename(image_path)
            })
            refresh_panel_graph()

            analysis_results.append({
                "drone":       drone_id,
                "zone":        zone,
                "panel_index": panel_index,
                "panel_id":    f"{zone}_p{panel_index}",
                "image":       image_path,
                "analysis":    analysis
            })

            if zone not in zone_panel_buffer:
                zone_panel_buffer[zone] = []
            zone_panel_buffer[zone].append({
                "panel_index": panel_index,
                "class":       analysis["class"],
                "confidence":  analysis["confidence"]
            })

            # After panel 4: write zone summary, mark complete
            if panel_index == 4:
                panels = zone_panel_buffer[zone]
                class_counts = Counter(p["class"] for p in panels)
                avg_conf = sum(p["confidence"] for p in panels) / len(panels)
                update_zone(zone, {
                    "class":           class_counts.most_common(1)[0][0],
                    "confidence":      round(avg_conf, 2),
                    "panel_breakdown": dict(class_counts),
                    "reasoning":       f"Panel breakdown: {dict(class_counts)}"
                })
                drone.complete_zone(zone)
                completed_zones.add(zone)
                logs.append(f"[ZONE_DONE] {drone_id} zone={zone}")
                log_event("executor", "zone_complete", {"zone": zone, "drone": drone_id})

            swarm_logs.setdefault(drone_id, []).append(f"panel {panel_index} {zone}")

        #  RETURN TO DOCK 
        elif action == "return_to_dock":
            drone.return_to_dock()
            logs.append(f"[DOCK] {drone_id} returned [recharged:100%]")
            swarm_logs.setdefault(drone_id, []).append("docked")

        else:
            logs.append(f"{drone_id}: unknown action {action}")

    return {
        "execution_log":  logs,
        "swarm_logs":     swarm_logs,
        "analysis":       analysis_results,
        "swarm_messages": all_swarm_msgs,
        "next_step":      "safety"
    }
