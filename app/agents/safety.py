from app.observability.tracer import log_event

CONFIDENCE_THRESHOLD = 0.55
CRITICAL_CLASSES     = {"Physical-Damage", "Electrical-Damage"}
ABORT_THRESHOLD      = 8    # panels (not zones)
ESCALATE_THRESHOLD   = 3    # panels

from langsmith import traceable

@traceable(name="safety_agent", run_type="chain")
def safety_agent(state):
    """
    Panel-level safety review.
    Flags individual panels, not zones.
    Abort only if many panels show critical damage at high confidence.
    """
    analysis = state.get("analysis", [])
    flags    = []
    critical_count = 0

    for result in analysis:
        a           = result.get("analysis", {})
        zone        = result.get("zone", "unknown")
        panel_index = result.get("panel_index", "?")
        panel_id    = f"{zone}_p{panel_index}"
        confidence  = a.get("confidence", 0)
        cls         = a.get("class", "Unknown")

        if cls in ("mock", "Unknown"):
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            flags.append({
                "panel":  panel_id,
                "zone":   zone,
                "reason": "low_confidence",
                "confidence": confidence,
                "class":  cls
            })

        if cls in CRITICAL_CLASSES and confidence > 0.7:
            critical_count += 1
            flags.append({
                "panel":  panel_id,
                "zone":   zone,
                "reason": "critical_damage_detected",
                "confidence": confidence,
                "class":  cls
            })

    log_event("safety", "flags",          flags)
    log_event("safety", "critical_panels", critical_count)

    if critical_count >= ABORT_THRESHOLD:
        log_event("safety", "decision", "aborted")
        return {"safety_status": "aborted",   "safety_flags": flags, "next_step": "reflection"}

    if critical_count >= ESCALATE_THRESHOLD or \
       sum(1 for f in flags if f["reason"] == "low_confidence") >= 3:
        log_event("safety", "decision", "escalated")
        return {"safety_status": "escalated", "safety_flags": flags, "next_step": "reflection"}

    log_event("safety", "decision", "approved")
    return {"safety_status": "approved",  "safety_flags": [],  "next_step": "reflection"}