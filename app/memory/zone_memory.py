import json
import os

MEMORY_FILE = "zone_memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def get_zone(zone):
    return load_memory().get(zone, [])


def update_zone(zone, result):
    memory = load_memory()
    if zone not in memory:
        memory[zone] = []
    memory[zone].append(result)
    save_memory(memory)