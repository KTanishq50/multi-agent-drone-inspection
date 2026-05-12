import networkx as nx

G = nx.grid_2d_graph(8, 8)


def get_neighbors(zone):
    try:
        i, j = map(int, zone.replace("zone_", "").split("_"))
        return [f"zone_{x}_{y}" for x, y in G.neighbors((i, j))]
    except Exception:
        return []