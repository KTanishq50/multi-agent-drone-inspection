"""
app/memory/panel_graph.py

Panel-level spatial graph.

Each zone has 5 panels arranged left-to-right: p0 p1 p2 p3 p4
Panel adjacency:
  - Within a zone: p0-p1-p2-p3-p4 (horizontal strip)
  - Across zones: panels on the edge of one zone connect to
    panels on the edge of adjacent zones.

    zone_i_j:   p0 p1 p2 p3 p4
    zone_i_j+1: p0 p1 p2 p3 p4
    
    Right edge of zone_i_j (p4) → Left edge of zone_i_j+1 (p0)
    Bottom edge panels (all 5) → Top edge panels of zone_i+1_j

This gives a connected panel graph across the entire 8x8 farm.
"""
import networkx as nx

_G = None


def _build_panel_graph():
    global _G
    _G = nx.Graph()

    ROWS, COLS, PANELS = 8, 8, 5

    # Add all panel nodes
    for i in range(ROWS):
        for j in range(COLS):
            for p in range(PANELS):
                pid = f"zone_{i}_{j}_p{p}"
                _G.add_node(pid, zone=f"zone_{i}_{j}", panel_index=p, row=i, col=j)

    # Within-zone edges: panels are adjacent to each other horizontally
    for i in range(ROWS):
        for j in range(COLS):
            for p in range(PANELS - 1):
                _G.add_edge(
                    f"zone_{i}_{j}_p{p}",
                    f"zone_{i}_{j}_p{p+1}",
                    type="within_zone"
                )

    # Cross-zone edges: right edge of zone → left edge of right neighbour
    # p4 of zone_i_j ↔ p0 of zone_i_j+1
    for i in range(ROWS):
        for j in range(COLS - 1):
            _G.add_edge(
                f"zone_{i}_{j}_p4",
                f"zone_{i}_{j+1}_p0",
                type="cross_zone_horizontal"
            )

    # Cross-zone edges: bottom panels of zone → top panels of zone below
    # All 5 panels of zone_i_j ↔ corresponding panels of zone_i+1_j
    for i in range(ROWS - 1):
        for j in range(COLS):
            for p in range(PANELS):
                _G.add_edge(
                    f"zone_{i}_{j}_p{p}",
                    f"zone_{i+1}_{j}_p{p}",
                    type="cross_zone_vertical"
                )

    return _G


def get_panel_graph():
    global _G
    if _G is None:
        _build_panel_graph()
    return _G


def get_panel_neighbors(zone, panel_index):
    """Return list of (zone, panel_index) tuples adjacent to this panel."""
    G = get_panel_graph()
    pid = f"{zone}_p{panel_index}"
    if pid not in G:
        return []
    neighbors = []
    for n in G.neighbors(pid):
        node_data = G.nodes[n]
        neighbors.append({
            "panel_id": n,
            "zone":     node_data.get("zone"),
            "panel_index": node_data.get("panel_index"),
            "edge_type": G.edges[pid, n].get("type", "unknown")
        })
    return neighbors