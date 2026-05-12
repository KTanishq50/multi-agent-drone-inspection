from app.observability.tracer import log_event

MAX_ITERATIONS = 3

from langsmith import traceable

@traceable(name="supervisor_agent", run_type="chain")
def supervisor_agent(state):
    """
    Dynamic supervisor / orchestrator.

    Responsibilities:
    1. On first entry: decompose mission into zone assignments,
       communicate drone assignments via swarm_messages
    2. After report returns: check mission score and decide to end or replan
    3. Tracks which zones were completed vs abandoned (battery forced return)
       and flags incomplete zones for reassignment in next iteration

    Agent-to-agent communication model:
    - Supervisor broadcasts zone assignments to drones via supervisor_messages
      in the state (planner reads these)
    - Planner builds the execution plan based on supervisor assignments
    - Executor reports back via analysis results
    - Supervisor reads mission_score after reflection and decides next step
    """
    iteration     = state.get("iteration", 0)
    mission_score = state.get("mission_score", None)
    report        = state.get("report", None)
    analysis      = state.get("analysis", [])
    execution_log = state.get("execution_log", [])

    log_event("supervisor", "iteration", iteration)

    # ── First entry: start mission ─────────────────────────────────────────
    if report is None and iteration == 0:
        log_event("supervisor", "decision", "plan — mission start")
        return {
            "supervisor_decision": "plan",
            "iteration":           iteration + 1,
            "supervisor_notes":    "Initial mission assignment. Drones start at dock with full battery."
        }

    # ── After first execution: check for incomplete zones ─────────────────
    if report is None and iteration > 0:
        # Check if any zone was abandoned due to battery
        battery_critical = [
            line for line in execution_log
            if "[BATTERY CRITICAL]" in line
        ]
        incomplete_zones = []
        for line in execution_log:
            import re
            m = re.search(r"\[BATTERY CRITICAL\] (drone_\d+)", line)
            if m:
                # Find what zone this drone was on when battery hit
                drone = m.group(1)
                incomplete_zones.append(drone)

        if battery_critical:
            log_event("supervisor", "battery_critical_detected", battery_critical)
            log_event("supervisor", "reassigning_incomplete_zones", incomplete_zones)
            return {
                "supervisor_decision": "plan",
                "iteration":           iteration + 1,
                "supervisor_notes": (
                    f"Battery critical detected on iteration {iteration}. "
                    f"Supervisor reassigning incomplete zones to available drone. "
                    f"Affected drones: {', '.join(incomplete_zones)}"
                )
            }

        log_event("supervisor", "decision", "plan — continuing")
        return {
            "supervisor_decision": "plan",
            "iteration":           iteration + 1,
            "supervisor_notes":    "Continuing mission."
        }

    # ── After report: evaluate and decide ─────────────────────────────────
    if report is not None:
        # Check panels that had low confidence — flag for re-inspection
        low_conf_panels = []
        for a in analysis:
            if a.get("analysis", {}).get("confidence", 1.0) < 0.60:
                low_conf_panels.append(a.get("panel_id", "?"))

        if low_conf_panels:
            log_event("supervisor", "low_confidence_panels", low_conf_panels)

        # Replan if score is low and iterations remain
        if mission_score is not None and mission_score < 0.5 and iteration < MAX_ITERATIONS:
            log_event("supervisor", "decision",
                      f"replan — score {mission_score:.2f} below threshold")
            return {
                "supervisor_decision": "plan",
                "iteration":           iteration + 1,
                "report":              None,
                "supervisor_notes": (
                    f"Mission score {mission_score:.2f} below 0.5. "
                    f"Replanning. Low confidence panels: {low_conf_panels}. "
                    f"Supervisor instructing planner to prioritize these panels."
                )
            }

        log_event("supervisor", "decision", f"end — score {mission_score:.2f}")
        return {
            "supervisor_decision": "end",
            "supervisor_notes": (
                f"Mission complete. Score: {mission_score:.2f}. "
                f"Iterations used: {iteration}."
            )
        }

    log_event("supervisor", "decision", "end — fallback")
    return {"supervisor_decision": "end"}