import re
from app.memory.zone_memory import get_zone
from app.memory.zone_graph import get_neighbors
from app.memory.panel_memory import get_panel
from app.memory.panel_feedback_store import get_panel_feedback, get_high_uncertainty_panels
from app.memory.panel_graph import get_panel_neighbors
from app.observability.tracer import log_event


def extract_zones(text):
    return re.findall(r"zone_\d+_\d+", text)

from langsmith import traceable

@traceable(name="planner_agent", run_type="chain")
def planner_agent(state):
    """
    Panel-aware planner. Reads supervisor_notes for context.

    Zone assignment logic:
    - drone_1 gets odd-indexed zones (by risk rank), drone_2 gets even
    - Both drones assigned same zone so swarm skip fires
    - Within each zone, panels are scanned in natural order P0→P4
      (risk-based reordering caused log confusion, removed)
    - High-uncertainty panels from feedback get their zone added automatically
    - Supervisor notes about battery/reassignment are logged
    """
    user_input       = state["user_input"]
    supervisor_notes = state.get("supervisor_notes", "")
    zones            = extract_zones(user_input)

    log_event("planner", "zones_extracted", zones)
    if supervisor_notes:
        log_event("planner", "supervisor_notes", supervisor_notes)

    if not zones:
        return {"plan": [], "error": "No zones found", "next_step": "end"}

    drones = ["drone_1", "drone_2"]

    # Add high-uncertainty zones from panel feedback
    uncertain_panels = get_high_uncertainty_panels(threshold=0.35)
    for up in uncertain_panels:
        pid   = up["panel_id"]
        parts = pid.rsplit("_p", 1)
        if len(parts) == 2:
            uz = parts[0]
            if uz not in zones:
                zones.append(uz)
                log_event("planner", "added_uncertain_zone", {
                    "zone":  uz,
                    "panel": pid,
                    "score": up["disagreement_score"]
                })

    # ── Panel-level risk scoring ──────────────────────────────────────────
    zone_risk_scores = []

    for zone in zones:
        zone_risk = 0.0

        for p_idx in range(5):
            panel_risk = 0.0

            # Panel's own history
            history = get_panel(zone, p_idx)
            if history:
                last = history[-1]
                if isinstance(last, dict):
                    cls  = last.get("class", "")
                    conf = last.get("confidence", 0)
                    if cls in ("Physical-Damage", "Electrical-Damage"):
                        panel_risk += conf * 1.5
                    elif cls in ("Bird-drop", "Dusty"):
                        panel_risk += conf * 0.8
                    elif cls == "Snow-Covered":
                        panel_risk += conf * 0.5

            # Panel feedback disagreement
            feedback = get_panel_feedback(zone, p_idx)
            if feedback:
                panel_risk += feedback.get("disagreement_score", 0) * 0.4
                panel_risk += feedback.get("avg_calibration_error", 0) * 0.3

            # Adjacent panel damage history
            for neighbour in get_panel_neighbors(zone, p_idx):
                n_history = get_panel(neighbour["zone"], neighbour["panel_index"])
                if n_history:
                    n_last = n_history[-1]
                    if isinstance(n_last, dict):
                        n_cls  = n_last.get("class", "")
                        n_conf = n_last.get("confidence", 0)
                        if n_cls in ("Physical-Damage", "Electrical-Damage"):
                            panel_risk += n_conf * 0.6
                        elif n_cls == "Bird-drop":
                            panel_risk += n_conf * 0.3

            zone_risk += panel_risk

        zone_risk_scores.append((zone, zone_risk))

    # Sort zones by risk — highest first
    zone_risk_scores.sort(key=lambda x: x[1], reverse=True)
    log_event("planner", "zone_priority",
              [(z, round(r, 3)) for z, r in zone_risk_scores])

    plan = []

    for i, (zone, zone_risk) in enumerate(zone_risk_scores):
        primary   = drones[i % 2]
        secondary = drones[(i + 1) % 2]

        # Supervisor communication: log which drone is assigned which zone
        log_event("planner", "zone_assignment", {
            "zone":      zone,
            "primary":   primary,
            "secondary": secondary,
            "risk":      round(zone_risk, 3),
            "note":      f"Supervisor assigned {zone} to {primary} (risk={zone_risk:.2f}). "
                         f"{secondary} assigned as backup — will skip via swarm if {primary} completes first."
        })

        # Primary drone: move → P0,P1,P2,P3,P4 in ORDER (no risk reordering)
        # Scanning in natural order keeps log readable and animation correct
        plan.append({"drone": primary, "action": "move_to", "zone": zone})
        for p_idx in range(5):
            plan.append({
                "drone":       primary,
                "action":      "capture_panel",
                "zone":        zone,
                "panel_index": p_idx
            })
        plan.append({"drone": primary, "action": "return_to_dock", "zone": zone})

        # Secondary drone: assigned same zone → will skip after swarm message
        plan.append({"drone": secondary, "action": "move_to",       "zone": zone})
        plan.append({"drone": secondary, "action": "capture_panel",
                     "zone": zone, "panel_index": 0})
        plan.append({"drone": secondary, "action": "return_to_dock", "zone": zone})

    log_event("planner", "total_tasks", len(plan))
    return {"plan": plan, "next_step": "execute"}