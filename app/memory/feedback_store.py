import json
import os

FEEDBACK_FILE = "feedback_signals.json"


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return {}
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)


def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def store_feedback_signal(zone, signal):
    data = load_feedback()

    if zone not in data:
        data[zone] = []
    data[zone].append(signal)

    zone_signals = data[zone]
    disagreements = sum(1 for s in zone_signals if s.get("disagreement"))
    avg_conf = sum(s.get("confidence", 0) for s in zone_signals) / len(zone_signals)

    data[f"{zone}_meta"] = {
        "total_missions": len(zone_signals),
        "disagreement_score": disagreements / len(zone_signals),
        "avg_confidence": avg_conf,
        "avg_calibration_error": abs(0.8 - avg_conf)
    }

    save_feedback(data)


def get_feedback_for_zone(zone):
    data = load_feedback()
    return data.get(f"{zone}_meta", None)