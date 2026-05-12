"""
app/memory/panel_memory.py

Panel-level episodic memory.
Each panel has its own history: zone_0_0_p0, zone_0_0_p1, ... zone_7_7_p4

Replaces zone_memory as the primary memory store for intelligence.
zone_memory is still used for zone-level summaries (for backward compat).
"""
import json
import os

PANEL_MEMORY_FILE = "panel_memory.json"


def panel_id(zone, panel_index):
    return f"{zone}_p{panel_index}"


def load_panel_memory():
    if not os.path.exists(PANEL_MEMORY_FILE):
        return {}
    with open(PANEL_MEMORY_FILE, "r") as f:
        return json.load(f)


def save_panel_memory(data):
    with open(PANEL_MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_panel(zone, panel_index):
    """Return history list for a specific panel."""
    pid = panel_id(zone, panel_index)
    return load_panel_memory().get(pid, [])


def update_panel(zone, panel_index, result):
    """Store a finding for a specific panel."""
    pid = panel_id(zone, panel_index)
    data = load_panel_memory()
    if pid not in data:
        data[pid] = []
    data[pid].append(result)
    save_panel_memory(data)


def get_all_panels():
    """Return all panel histories."""
    return load_panel_memory()