from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import (
    supervisor_node,
    planner_node,
    executor_node,
    safety_node,
    reflection_node,
    report_node
)


def route_supervisor(state):
    decision = state.get("supervisor_decision", "plan")
    if decision == "end":
        return END
    return "planner"


def route_safety(state):
    """
    ALL paths go through reflection now so mission_score is always computed.
    Safety status (aborted/escalated/approved) is read by reflection and report.
    """
    return "reflection"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner",    planner_node)
    builder.add_node("executor",   executor_node)
    builder.add_node("safety",     safety_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("report_agent",     report_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor
    )

    builder.add_edge("planner",   "executor")
    builder.add_edge("executor",  "safety")

    # Safety always goes to reflection — reflection reads safety_status
    builder.add_conditional_edges(
        "safety",
        route_safety,
        {"reflection": "reflection"}
    )

    builder.add_edge("reflection", "report_agent")
    builder.add_edge("report_agent",     "supervisor")

    return builder.compile()