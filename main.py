import os
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from app.graph.builder import build_graph
from app.rag.ingest import ingest_documents
from app.memory.graph_rag import refresh_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "solar_farm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("chroma_db"):
        print("[startup] Building Chroma vector store...")
        ingest_documents()
    yield


app = FastAPI(title="Agentic Drone Inspection System", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
graph = build_graph()


class MissionRequest(BaseModel):
    user_input: str


@app.get("/", response_class=HTMLResponse)
def root():
    html = open(os.path.join(BASE_DIR, "ui.html")).read()
    return HTMLResponse(content=html)


@app.get("/grid-image/{zone}/{index}")
def get_zone_image(zone: str, index: int):
    zone_path = os.path.join(DATA_DIR, zone)
    if not os.path.exists(zone_path):
        return {"error": "zone not found"}
    files = sorted(
        f for ext in ["*.jpg", "*.jpeg", "*.png"]
        for f in glob.glob(os.path.join(zone_path, ext))
    )
    if not files or index >= len(files):
        return {"error": "image not found"}
    return FileResponse(files[index])


@app.post("/run")
def run_mission(req: MissionRequest):
    initial_state = {
        "user_input":          req.user_input,
        "plan":                [],
        "execution_log":       [],
        "drone_position":      "dock",
        "images_captured":     [],
        "analysis":            [],
        "next_step":           "",
        "safety_status":       "",
        "safety_flags":        [],
        "mission_score":       0.0,
        "feedback_signal":     [],
        "report":              None,
        "supervisor_decision": "",
        "supervisor_notes":    "",
        "iteration":           0,
        "swarm_messages":      []
    }
    result = graph.invoke(initial_state)
    refresh_graph()
    return {
        "report":             result.get("report"),
        "mission_score":      result.get("mission_score"),
        "safety_status":      result.get("safety_status"),
        "needs_human_review": result.get("safety_status") in ("escalated", "aborted"),
        "safety_flags":       result.get("safety_flags", []),
        "analysis":           result.get("analysis", []),
        "execution_log":      result.get("execution_log", []),
        "iterations":         result.get("iteration", 1),
        "swarm_messages":     result.get("swarm_messages", []),
        "supervisor_notes":   result.get("supervisor_notes", "")
    }


# ── Panel-level memory endpoints ──────────────────────────────────────────────

@app.get("/panel/{zone}/{panel_index}")
def get_panel_history(zone: str, panel_index: int):
    from app.memory.panel_memory import get_panel
    return {"panel_id": f"{zone}_p{panel_index}",
            "history": get_panel(zone, panel_index)}


@app.get("/panel-feedback/{zone}/{panel_index}")
def get_panel_feedback_endpoint(zone: str, panel_index: int):
    from app.memory.panel_feedback_store import get_panel_feedback
    return {"panel_id": f"{zone}_p{panel_index}",
            "feedback": get_panel_feedback(zone, panel_index)}


@app.get("/panel-graph/{zone}/{panel_index}")
def get_panel_graph_insights(zone: str, panel_index: int):
    from app.memory.panel_graph_rag import query_panel_graph
    return {"panel_id": f"{zone}_p{panel_index}",
            "insights": query_panel_graph(zone, panel_index)}


@app.get("/uncertain-panels")
def get_uncertain_panels():
    from app.memory.panel_feedback_store import get_high_uncertainty_panels
    return {"uncertain_panels": get_high_uncertainty_panels()}


# ── Zone-level endpoints (backward compat) ────────────────────────────────────

@app.get("/memory/{zone}")
def get_zone_history(zone: str):
    from app.memory.zone_memory import get_zone
    return {"zone": zone, "history": get_zone(zone)}


@app.get("/feedback/{zone}")
def get_zone_feedback(zone: str):
    from app.memory.feedback_store import get_feedback_for_zone
    return {"zone": zone, "feedback": get_feedback_for_zone(zone)}


@app.get("/graph/insights/{zone}")
def get_graph_insights(zone: str):
    from app.memory.graph_rag import query_graph
    return {"zone": zone, "insights": query_graph(zone)}