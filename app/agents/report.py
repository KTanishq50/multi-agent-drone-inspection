from app.core.llm import get_llm
from app.observability.tracer import log_event
from collections import defaultdict

from langsmith import traceable

@traceable(name="report_agent", run_type="chain")
def report_agent(state):
    analysis      = state.get("analysis", [])
    safety_status = state.get("safety_status", "approved")
    safety_flags  = state.get("safety_flags", [])
    mission_score = state.get("mission_score", 0.0)

    log_event("report", "generating", {"safety_status": safety_status})

    if not analysis:
        return {"report": "No data available.", "next_step": "end"}

    # Group panels by zone — zones are just grouping, not verdicts
    by_zone = defaultdict(list)
    for a in analysis:
        by_zone[a.get("zone", "?")].append({
            "panel_id":   a.get("panel_id", "?"),
            "panel":      a.get("panel_index", "?"),
            "drone":      a.get("drone", "?"),
            "class":      a.get("analysis", {}).get("class", "?"),
            "confidence": a.get("analysis", {}).get("confidence", 0),
            "meta_note":  a.get("analysis", {}).get("meta_note", ""),
        })

    zone_sections = [
        {
            "zone": z,
            "panels": sorted(panels, key=lambda p: p["panel"])
        }
        for z, panels in by_zone.items()
    ]

    abort_note = ""
    if safety_status == "aborted":
        abort_note = "MISSION ABORTED — Critical damage threshold exceeded."
    elif safety_status == "escalated":
        abort_note = "HUMAN REVIEW REQUESTED — Flags raised."

    llm = get_llm()
    prompt = f"""
You are generating a solar farm inspection report.
The farm has 8x8 zones. Each zone has 5 individual solar panels (P0-P4).
EACH PANEL WAS INSPECTED INDEPENDENTLY. Zones are only grouping containers.

Mission score: {mission_score:.2f}
Safety status: {safety_status.upper()}
{abort_note}

Panel findings by zone:
{zone_sections}

Safety flags (panel-level):
{safety_flags}

Generate a report with:
1. EXECUTIVE SUMMARY
2. PANEL-LEVEL FINDINGS
   For each zone: show each panel (P0-P4) with its individual condition.
   Do NOT assign a single verdict to the zone — each panel stands alone.
3. CRITICAL PANELS (confidence > 0.7 and damage class)
4. DAMAGE PATTERNS (any panels with same defect sharing adjacency)
5. SAFETY FLAGS & HITL RECOMMENDATIONS
6. RECOMMENDED ACTIONS per panel

Be concise. Panels within a zone may have different conditions.
Do not merge or aggregate panel results.
"""

    response = llm.invoke(prompt)
    return {"report": response.content, "next_step": "end"}