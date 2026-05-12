"""
app/memory/panel_feedback_store.py

Panel-level feedback store.
Tracks per-panel calibration across missions:
  - How often fast/slow classifiers disagreed on this panel
  - Average confidence
  - Which meta-reasoner path was taken most
  - DPO pairs for future fine-tuning

Planner reads this to boost priority of panels where the system was uncertain.
"""
import json
import os

PANEL_FEEDBACK_FILE = "panel_feedback.json"


def load_panel_feedback():
    if not os.path.exists(PANEL_FEEDBACK_FILE):
        return {}
    with open(PANEL_FEEDBACK_FILE, "r") as f:
        return json.load(f)


def save_panel_feedback(data):
    with open(PANEL_FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def store_panel_feedback(zone, panel_index, signal):
    """
    Store a feedback signal for a panel.
    signal keys: class, confidence, disagreement, meta_note, reasoning,
                 dpo_preferred, dpo_context
    """
    pid = f"{zone}_p{panel_index}"
    data = load_panel_feedback()

    if pid not in data:
        data[pid] = []
    data[pid].append(signal)

    # Update aggregate meta
    signals = data[pid]
    disagreements = sum(1 for s in signals if s.get("disagreement"))
    avg_conf = sum(s.get("confidence", 0) for s in signals) / len(signals)
    class_counts = {}
    for s in signals:
        c = s.get("class", "Unknown")
        class_counts[c] = class_counts.get(c, 0) + 1
    most_common = max(class_counts, key=class_counts.get) if class_counts else "Unknown"

    data[f"{pid}_meta"] = {
        "total_inspections":  len(signals),
        "disagreement_score": disagreements / len(signals),
        "avg_confidence":     avg_conf,
        "avg_calibration_error": abs(0.85 - avg_conf),
        "most_common_class":  most_common,
        "class_history":      class_counts
    }

    save_panel_feedback(data)


def get_panel_feedback(zone, panel_index):
    """Return aggregate meta for a panel, or None."""
    pid = f"{zone}_p{panel_index}"
    data = load_panel_feedback()
    return data.get(f"{pid}_meta", None)


def get_high_uncertainty_panels(threshold=0.3):
    """
    Return list of panel IDs where disagreement_score > threshold.
    Planner uses this to flag panels needing re-inspection.
    """
    data = load_panel_feedback()
    uncertain = []
    for key, val in data.items():
        if key.endswith("_meta") and isinstance(val, dict):
            if val.get("disagreement_score", 0) > threshold:
                panel_id = key.replace("_meta", "")
                uncertain.append({
                    "panel_id": panel_id,
                    "disagreement_score": val["disagreement_score"],
                    "avg_confidence": val["avg_confidence"],
                    "most_common_class": val["most_common_class"]
                })
    return sorted(uncertain, key=lambda x: x["disagreement_score"], reverse=True)