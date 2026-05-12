from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    user_input: str
    plan: List[Dict]
    execution_log: List[str]
    drone_position: str
    images_captured: List[str]
    analysis: List[Dict]
    next_step: str
    safety_status: str
    safety_flags: List[Dict]
    mission_score: float
    feedback_signal: List[Dict]
    report: Optional[str]
    supervisor_decision: str
    supervisor_notes: str        # NEW: supervisor broadcasts to agents
    iteration: int
    swarm_messages: List[Dict]