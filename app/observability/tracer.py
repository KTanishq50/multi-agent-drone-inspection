import json
import time
import os

TRACE_FILE = "trace_logs.json"


def log_event(agent, step, data):
    trace = {
        "timestamp": time.time(),
        "agent": agent,
        "step": step,
        "data": data
    }

    if not os.path.exists(TRACE_FILE):
        logs = []
    else:
        with open(TRACE_FILE, "r") as f:
            logs = json.load(f)

    logs.append(trace)

    with open(TRACE_FILE, "w") as f:
        json.dump(logs, f, indent=2)