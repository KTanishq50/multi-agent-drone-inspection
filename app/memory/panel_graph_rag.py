"""
app/memory/panel_graph_rag.py

Panel-level GraphRAG.

Builds a knowledge graph where:
  - Nodes: individual panels (zone_i_j_p0 ... p4) + defect class nodes
  - Edges:
      panel → defect (from panel_memory history)
      panel → panel  (spatial adjacency from panel_graph)

query_panel_graph(zone, panel_index) returns contextual insights:
  - What defects this panel has had in past missions
  - What defects its immediate neighbours have had
  - Whether a damage corridor exists (multiple adjacent panels damaged)
"""
import networkx as nx
from app.memory.panel_memory import load_panel_memory
from app.memory.panel_graph import get_panel_neighbors
from app.observability.tracer import log_event

from langsmith import traceable
_G = nx.Graph()
_graph_built = False


def _build_graph():
    global _G, _graph_built
    panel_memory = load_panel_memory()

    _G.clear()

    # Add panel nodes from memory
    for pid in panel_memory:
        _G.add_node(pid, type="panel")

    # Add defect nodes and edges from history
    for pid, records in panel_memory.items():
        for r in records:
            if isinstance(r, dict):
                cls = r.get("class")
                if cls and cls not in ("Unknown", "mock"):
                    defect_node = f"defect_{cls}"
                    _G.add_node(defect_node, type="defect")
                    # Weight = confidence of this finding
                    _G.add_edge(
                        pid, defect_node,
                        weight=r.get("confidence", 0.5),
                        type="panel_defect"
                    )

    # Add spatial adjacency edges between panels
    # Parse panel IDs: zone_i_j_pN
    import re
    for pid in list(_G.nodes):
        if _G.nodes[pid].get("type") != "panel":
            continue
        m = re.match(r"(zone_\d+_\d+)_p(\d+)$", pid)
        if not m:
            continue
        zone = m.group(1)
        p_idx = int(m.group(2))
        for neighbour in get_panel_neighbors(zone, p_idx):
            n_pid = neighbour["panel_id"]
            if n_pid not in _G:
                _G.add_node(n_pid, type="panel")
            if not _G.has_edge(pid, n_pid):
                _G.add_edge(pid, n_pid, type=neighbour["edge_type"])

    _graph_built = True


def refresh_panel_graph():
    global _graph_built
    _graph_built = False

@traceable(name="panel_graph_rag_query", run_type="retriever")
def query_panel_graph(zone, panel_index):
    """
    Returns a list of insight strings for the given panel.
    Used by perception_agent to enrich the LLM prompt.
    """
    global _graph_built
    if not _graph_built:
        _build_graph()

    pid = f"{zone}_p{panel_index}"

    # Ensure panel node exists even if no history yet
    if pid not in _G:
        # Still build spatial context from neighbours in memory
        insights = []
        for neighbour in get_panel_neighbors(zone, panel_index):
            n_pid = neighbour["panel_id"]
            if n_pid in _G:
                for nn in _G.neighbors(n_pid):
                    if _G.nodes.get(nn, {}).get("type") == "defect":
                        cls = nn.replace("defect_", "")
                        edge_type = neighbour["edge_type"]
                        insights.append(
                            f"Neighbour panel {n_pid} "
                            f"({edge_type.replace('_', ' ')}) "
                            f"has history of {cls}"
                        )
        return insights

    insights = []

    # Direct defect history for this panel
    for neighbour in _G.neighbors(pid):
        node_data = _G.nodes.get(neighbour, {})
        if node_data.get("type") == "defect":
            cls = neighbour.replace("defect_", "")
            conf = _G.edges[pid, neighbour].get("weight", 0)
            insights.append(
                f"This panel has previous history of {cls} "
                f"(confidence {conf:.2f})"
            )

    # Neighbour panel defect history
    damage_corridor = []
    for neighbour in _G.neighbors(pid):
        n_data = _G.nodes.get(neighbour, {})
        if n_data.get("type") != "panel":
            continue
        edge_type = _G.edges[pid, neighbour].get("type", "")
        for nn in _G.neighbors(neighbour):
            nn_data = _G.nodes.get(nn, {})
            if nn_data.get("type") == "defect":
                cls = nn.replace("defect_", "")
                conf = _G.edges[neighbour, nn].get("weight", 0)
                rel = edge_type.replace("_", " ")
                insights.append(
                    f"Adjacent panel {neighbour} ({rel}) "
                    f"has history of {cls} (confidence {conf:.2f})"
                )
                if "Damage" in cls or "Bird" in cls:
                    damage_corridor.append(neighbour)

    # Damage corridor detection
    if len(damage_corridor) >= 2:
        insights.append(
            f"DAMAGE CORRIDOR DETECTED: {len(damage_corridor)} adjacent panels "
            f"show damage history — elevated risk for this panel"
        )

    log_event("panel_graph_rag", "query", {
        "panel": pid,
        "insights_count": len(insights)
    })

    return insights