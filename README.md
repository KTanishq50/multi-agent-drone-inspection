# Multi Agent Drone Inspection System

A stateful multi-agent drone inspection platform for autonomous solar farm monitoring built with LangGraph, FastAPI, OpenCV, Retrieval-Augmented Generation (RAG), persistent memory systems, and collaborative swarm coordination.

This project simulates a fleet of intelligent inspection drones capable of:

* Coordinated multi-agent task execution
* Autonomous mission planning and replanning
* Panel-level defect analysis
* Persistent operational memory across missions
* Swarm communication between drone agents
* Hybrid perception using OpenCV + LLM reasoning
* Safety escalation workflows
* Human-readable mission reporting
* Real-time mission visualization through a web UI

The system was designed as an end-to-end agentic architecture rather than a single-model demo. Every mission flows through a stateful orchestration graph where multiple specialized agents collaborate, exchange information, adapt to uncertainty, and improve future missions using accumulated memory.

---

# Demo Video


https://github.com/user-attachments/assets/4994c454-cf9e-4701-8b51-20c5c38db8aa



---

---

# Screenshots

> Add screenshots/gifs here



---

# Why This Project Exists

Large solar farms contain thousands of panels distributed across wide geographic areas. Manual inspection is slow, expensive, and operationally inefficient.

This project explores how an autonomous drone swarm can:

* Inspect solar infrastructure
* Detect visual defects
* Coordinate tasks between agents
* Maintain long-term operational memory
* Adapt future missions based on past uncertainty
* Escalate dangerous findings automatically

The focus of the project is not only computer vision.

The core engineering challenge addressed here is:

> How do you build a persistent, stateful, agentic system where multiple AI agents collaborate, reason, remember, communicate, and continuously improve over time?

---

# High-Level Architecture

```text
User Command
     |
     v
FastAPI API (/run)
     |
     v
LangGraph Stateful Orchestration Graph

supervisor
    -> planner
        -> executor
            -> perception
                -> safety
                    -> reflection
                        -> report
                            -> supervisor (loop)
```

The system operates as a cyclic state machine.

Each mission:

1. Starts with a natural language command
2. Generates a coordinated drone execution plan
3. Executes inspections panel-by-panel
4. Runs hybrid visual reasoning
5. Applies safety analysis
6. Generates feedback signals
7. Stores persistent memory
8. Produces a mission report
9. Optionally replans future missions based on mission quality

---

# Core Features

## Multi-Agent Orchestration

The project uses a LangGraph `StateGraph` architecture where each node represents a specialized autonomous agent.

### Agents

| Agent      | Responsibility                              |
| ---------- | ------------------------------------------- |
| Supervisor | Controls mission lifecycle and replanning   |
| Planner    | Generates optimized inspection plans        |
| Executor   | Simulates drone movement and task execution |
| Perception | Performs panel-level defect reasoning       |
| Safety     | Escalates dangerous or uncertain findings   |
| Reflection | Computes feedback and learning signals      |
| Report     | Produces mission summaries                  |

The graph is stateful, meaning all agents share and modify a common mission state.

---

## Stateful Memory System

Unlike stateless demos, this system persists operational knowledge across missions.

### Persistent Memory Layers

| Memory                | Purpose                               |
| --------------------- | ------------------------------------- |
| `panel_memory.json`   | Historical panel inspection results   |
| `panel_feedback.json` | Uncertainty and disagreement tracking |
| `zone_memory.json`    | Zone-level summaries                  |
| `trace_logs.json`     | Full execution traces                 |
| `ChromaDB`            | Embedded RAG document storage         |

The planner reads previous mission history to prioritize risky or uncertain zones automatically.

This creates a feedback-driven self-improving inspection loop.

---

## Hybrid Perception Architecture

The perception system combines:

* Classical computer vision
* Retrieval-Augmented Generation (RAG)
* LLM reasoning
* Historical memory
* Meta-reasoning logic

Instead of relying entirely on a single deep learning model, the system uses a dual-process reasoning architecture.

### Fast Path (System 1)

OpenCV extracts visual features such as:

* Brightness
* Edge density
* Texture variance
* White pixel ratios
* Local roughness
* Color channel statistics

This produces fast deterministic classification signals.

### Slow Path (System 2)

An LLM receives:

* Visual statistics
* Historical panel memory
* RAG knowledge context
* Current mission information

The model performs contextual reasoning about defect types.

### Meta-Reasoner

A final arbitration layer decides:

* Which classifier to trust
* Whether confidence is sufficient
* Whether uncertainty should be escalated

This architecture was inspired by cognitive dual-process systems.

---

## Swarm Communication

Drone agents communicate through a shared message bus.

### Example swarm events

* `zone_done`
* `skip_zone`
* `battery_low`

This allows:

* Dynamic task reassignment
* Avoiding duplicate work
* Cooperative execution
* Battery-aware coordination

The swarm system simulates decentralized drone collaboration behavior.

---

## Autonomous Replanning

The supervisor agent evaluates:

* Mission score
* Safety outcomes
* Confidence metrics
* Reflection feedback

If the mission quality is low, the graph loops back into planning.

This transforms the system from:

> Static execution

into:

> Adaptive autonomous orchestration

---

## Safety-Critical Reasoning

The safety layer operates independently from perception.

It evaluates:

* Low-confidence predictions
* Electrical damage
* Physical damage
* Farm-wide critical thresholds

Possible outcomes:

* Approved
* Escalated
* Aborted

Even aborted missions continue into reflection so learning signals are still generated.

---

## Retrieval-Augmented Generation (RAG)

The system contains a lightweight domain-specific RAG pipeline.

### Knowledge Sources

* Solar inspection guides
* Defect documentation
* Technical operational notes

### Custom Embedding System

Instead of large generic embeddings, the project uses:

* TF-IDF style encoding
* Domain-specific vocabulary
* Lightweight vector representations
* ChromaDB vector storage

This keeps the system lightweight while preserving retrieval quality for a specialized domain.

---

# Technical Design

## Panel-First Intelligence Model

The architecture treats:

* Panels as primary intelligent entities
* Zones as organizational containers

Every panel maintains:

* Historical memory
* Confidence history
* Defect history
* Feedback signals
* Graph relationships

This enables fine-grained operational reasoning.

---

## Shared Agent State

All agents operate over a shared `AgentState` object.

Example state fields:

```python
class AgentState(TypedDict):
    user_input: str
    plan: List[Dict]
    execution_log: List[str]
    drone_position: str
    images_captured: List[str]
    analysis: List[Dict]
    safety_status: str
    mission_score: float
```

This architecture allows:

* Agent interoperability
* Persistent context propagation
* Multi-stage reasoning
* Stateful orchestration

---

## Observability and Tracing

The project includes two observability layers.

### Local Tracing

Every major event is logged into structured JSON traces.

Tracked events include:

* Planning decisions
* Perception outputs
* Safety escalations
* Swarm messages
* Final classifications

### LangSmith Integration

The system also supports LangSmith tracing for:

* Graph visualization
* Agent execution tracking
* Debugging orchestration flows
* Monitoring state transitions

---

# Tech Stack

## Backend

* Python
* FastAPI
* LangGraph
* LangChain
* Uvicorn

## AI / Agent Systems

* Groq LLM APIs
* LangGraph StateGraph
* Multi-agent orchestration
* RAG pipelines

## Computer Vision

* OpenCV
* NumPy
* Pillow

## Data & Memory

* ChromaDB
* JSON persistence layers
* NetworkX graph structures

## Frontend

* HTML
* JavaScript
* CSS

## Observability

* LangSmith
* Custom JSON tracing

---

# File Structure

```text
agentic_drone_inspection_system/
|
+-- main.py
+-- ui.html
+-- .env
|
+-- app/
|   |
|   +-- agents/
|   +-- graph/
|   +-- memory/
|   +-- rag/
|   +-- tools/
|   +-- tracing/
|
+-- data/
|   +-- solar_farm/
|       +-- zone_0_0/
|       +-- zone_0_1/
|       +-- ...
|
+-- docs/
|   +-- solar_inspection_guide.txt
|   +-- defect_types.txt
|
+-- chroma_db/
|
+-- panel_memory.json
+-- panel_feedback.json
+-- zone_memory.json
+-- trace_logs.json
```

---

# Mission Workflow

## Example Mission

User command:

```text
Inspect zone_0_0 zone_1_1 zone_3_4
```

Execution flow:

1. FastAPI receives request
2. LangGraph initializes mission state
3. Supervisor starts mission
4. Planner assigns zones to drones
5. Executor simulates movement and captures panels
6. Perception agent analyzes images
7. Safety agent evaluates risk
8. Reflection computes learning signals
9. Report agent generates mission summary
10. Supervisor decides whether to end or replan

---

# Defect Categories

The simulation currently supports:

| Defect Type       | Characteristics                |
| ----------------- | ------------------------------ |
| Clean             | Normal brightness and texture  |
| Bird-drop         | Localized bright contamination |
| Dusty             | Uniform grey coverage          |
| Electrical-Damage | Hotspots and texture anomalies |
| Physical-Damage   | Cracks and high edge density   |
| Snow-Covered      | Large white surface coverage   |

---

# Engineering Challenges Solved

## Stateful Multi-Agent Coordination

Implemented a persistent orchestration loop where multiple agents collaboratively modify shared mission state.

## Agent Communication

Built peer-to-peer swarm messaging between drone agents.

## Hybrid Reasoning

Combined deterministic CV reasoning with probabilistic LLM reasoning.

## Persistent Operational Memory

Designed long-term memory structures for future mission adaptation.

## Autonomous Feedback Loops

Implemented reflection-driven replanning behavior.

## Safety-Critical Execution

Added escalation and abort mechanisms independent from perception.

## Lightweight RAG Infrastructure

Created a domain-specific embedding pipeline without relying on heavyweight embedding models.

---

# Example Use Cases

* Solar farm inspection
* Infrastructure monitoring
* Autonomous aerial inspection systems
* Multi-agent orchestration research
* Swarm robotics simulation
* Agent memory architectures
* AI observability experimentation
* Human-in-the-loop safety systems

---

# Running the Project

## Install Dependencies

```bash
pip install fastapi uvicorn langchain langgraph langchain-groq
pip install langchain-community langchain-text-splitters
pip install chromadb networkx opencv-python-headless numpy pillow
pip install langsmith python-dotenv
```

---

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-drone-inspector
```

---

## Build the RAG Index

```bash
python -m app.rag.ingest
```

---

## Start the Server

```bash
uvicorn main:app --reload
```

---

# Future Improvements

Potential extensions:

* Real drone telemetry integration
* Live video stream ingestion
* Reinforcement learning for route optimization
* Distributed agent execution
* ROS2 integration
* Real VLM integration
* Kubernetes deployment
* Edge deployment on Jetson hardware
* Real-time WebSocket telemetry
* Autonomous charging station scheduling

---

# What This Project Demonstrates

This project demonstrates practical understanding of:

* Agentic AI systems
* Multi-agent orchestration
* Stateful workflow architectures
* LangGraph internals
* RAG systems
* Persistent memory architectures
* Hybrid AI reasoning
* AI observability
* FastAPI backend systems
* Autonomous system design
* Distributed coordination patterns
* Human-in-the-loop safety workflows

---

# Project Status

This is an experimental research-oriented simulation project focused on exploring autonomous multi-agent inspection architectures.

The current implementation simulates drone operations in a controlled environment but the architecture is intentionally designed to be extensible toward real-world robotics integration.

---


---

# License

Add your preferred license here.
