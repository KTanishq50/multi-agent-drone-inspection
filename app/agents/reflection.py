from app.memory.panel_feedback_store import store_panel_feedback
from app.observability.tracer import log_event

from langsmith import traceable

@traceable(name="reflection_agent", run_type="chain")
def reflection_agent(state):
    """
    Panel-level reflection.
    Scores mission based on individual panel confidence.
    Stores per-panel feedback in panel_feedback_store.
    Planner reads this next mission to re-prioritize uncertain panels.
    """
    analysis = state.get("analysis", [])

    if not analysis:
        return {"mission_score": 0.0, "feedback_signal": [], "next_step": "report"}

    high_conf = 0
    total     = 0
    feedback_signals = []

    for a in analysis:
        result      = a.get("analysis", {})
        zone        = a.get("zone", "unknown")
        panel_index = a.get("panel_index", 0)

        confidence   = result.get("confidence", 0)
        defect_class = result.get("class", "Unknown")
        meta_note    = result.get("meta_note", "")
        reasoning    = result.get("reasoning", "")

        if defect_class in ("mock", "Unknown"):
            continue

        total += 1
        if confidence > 0.7:
            high_conf += 1

        # Disagreement = fast/slow didn't agree cleanly
        disagreement = meta_note not in (
            "both systems agreed",
            "filename + system agreed",
            ""
        )

        signal = {
            "zone":         zone,
            "panel_index":  panel_index,
            "class":        defect_class,
            "confidence":   confidence,
            "disagreement": disagreement,
            "meta_note":    meta_note,
            "reasoning":    reasoning,
            # DPO pair for future fine-tuning
            "dpo_preferred": reasoning,
            "dpo_context": (
                f"panel={zone}_p{panel_index}, "
                f"class={defect_class}, conf={confidence:.2f}, "
                f"meta={meta_note}"
            )
        }

        feedback_signals.append(signal)
        # Written to panel_feedback.json — planner reads on next mission
        store_panel_feedback(zone, panel_index, signal)

    mission_score = high_conf / total if total > 0 else 0.0

    log_event("reflection", "mission_score",    mission_score)
    log_event("reflection", "panels_scored",    total)
    log_event("reflection", "high_conf_panels", high_conf)
    log_event("reflection", "feedback_stored",  len(feedback_signals))

    return {
        "mission_score":   mission_score,
        "feedback_signal": feedback_signals,
        "next_step":       "report"
    }