import networkx as nx
from app.memory.zone_memory import load_memory
from app.observability.tracer import log_event

_G = nx.Graph()
_graph_built = False


def _build_graph():
    global _G, _graph_built
    memory = load_memory()

    _G.clear()

    for zone in memory:
        _G.add_node(zone, type="zone")

    for zone, records in memory.items():
        for r in records:
            if isinstance(r, dict):
                defect = r.get("class")
                if defect and defect not in ("mock", "Unknown"):
                    defect_node = f"defect_{defect}"
                    _G.add_node(defect_node, type="defect")
                    _G.add_edge(zone, defect_node, weight=r.get("confidence", 0.5))

    for i in range(8):
        for j in range(8):
            z = f"zone_{i}_{j}"
            if z not in _G:
                _G.add_node(z, type="zone")
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 8 and 0 <= nj < 8:
                    _G.add_edge(z, f"zone_{ni}_{nj}", type="spatial")

    _graph_built = True


def refresh_graph():
    """Call this after new zone findings are stored."""
    global _graph_built
    _graph_built = False

from langsmith import traceable

@traceable(name="zone_graph_rag_query", run_type="retriever")
def query_graph(zone):
    """Zone-aware GraphRAG query."""
    global _graph_built
    if not _graph_built:
        _build_graph()

    if zone not in _G:
        return []

    insights = []

    for neighbor in _G.neighbors(zone):
        data = _G.nodes[neighbor]
        if data.get("type") == "defect":
            defect_name = neighbor.replace("defect_", "")
            conf = _G.edges[zone, neighbor].get("weight", 0)
            insights.append(
                f"{zone} has history of {defect_name} (confidence {conf:.2f})"
            )

    for neighbor in _G.neighbors(zone):
        if _G.nodes[neighbor].get("type") == "zone":
            for nn in _G.neighbors(neighbor):
                if _G.nodes[nn].get("type") == "defect":
                    defect_name = nn.replace("defect_", "")
                    insights.append(
                        f"Adjacent zone {neighbor} has history of {defect_name}"
                    )

    log_event("graph_rag", "query", {"zone": zone, "insights": insights})
    return insights