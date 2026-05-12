from app.agents.supervisor import supervisor_agent
from app.agents.planner import planner_agent
from app.agents.executor import executor_agent
from app.agents.safety import safety_agent
from app.agents.reflection import reflection_agent
from app.agents.report import report_agent


def supervisor_node(state):
    return supervisor_agent(state)

def planner_node(state):
    return planner_agent(state)

def executor_node(state):
    return executor_agent(state)

def safety_node(state):
    return safety_agent(state)

def reflection_node(state):
    return reflection_agent(state)

def report_node(state):
    return report_agent(state)