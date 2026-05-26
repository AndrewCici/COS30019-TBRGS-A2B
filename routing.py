"""
routing.py
==========
Traffic-Based Route Guidance System (TBRGS) — routing engine.

Provides
--------
  calculate_travel_time(predicted_flow, distance_km) -> float (minutes)
  get_top_5_routes(origin, destination, model_name)  -> List[(path, minutes)]
"""

import os
import sys
import math
import heapq
import itertools
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sklearn.preprocessing import MinMaxScaler

from data_processing import TrafficDataProcessor
from models import TrafficLSTM, TrafficGRU, TrafficTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "scats_sample.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# ---------------------------------------------------------------------------
# Traffic-flow ↔ speed model constants  (Traffic Flow to Travel Time v1.0)
# ---------------------------------------------------------------------------
_A                  = -1.4648375   # quadratic coefficient
_B                  = 93.75        # linear coefficient
_FLOW_THRESHOLD     = 351.0        # veh/hr — speed capped below this flow
_SPEED_LIMIT        = 60.0         # km/h   — cap when flow ≤ 351
_INTERSECTION_DELAY = 0.5          # minutes added per intersection (node)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# REQUIREMENT 1 — Travel Time Calculation
# ===========================================================================

def calculate_travel_time(predicted_flow: float, distance_km: float) -> float:
    """
    Estimate travel time (minutes) for one road segment given predicted
    traffic flow and link distance.

    Model (from PDF):
        flow = A·speed² + B·speed    (A = -1.4648375,  B = 93.75)

    Speed derivation
    ----------------
    Rearranged: A·s² + B·s − flow = 0
    Quadratic solution:
        s = (−B ± √(B² + 4·A·flow)) / (2·A)

    Because A < 0, the *higher* root (green / under-capacity branch) is
    obtained by taking the '−' sign in the numerator:
        s_high = (−B − √disc) / (2·A)

    Rules (as specified):
    • flow ≤ 351 veh/hr → speed exceeds the speed limit; cap at 60 km/h.
    • flow >  351 veh/hr → solve for the higher root (under-capacity branch).

    Travel time:
        t = (distance_km / speed) × 60  +  0.5  (minutes, intersection delay)
    """
    if predicted_flow <= _FLOW_THRESHOLD:
        speed = _SPEED_LIMIT
    else:
        # Discriminant of A·s² + B·s − flow = 0
        # disc = B² − 4·A·(−flow) = B² + 4·A·flow
        disc = _B ** 2 + 4.0 * _A * predicted_flow
        disc = max(disc, 0.0)                        # guard at/past capacity
        sqrt_disc = math.sqrt(disc)
        # Higher root (green curve, under capacity)
        speed = (-_B - sqrt_disc) / (2.0 * _A)
        speed = max(speed, 1.0)                      # hard floor — no div-by-zero

    return (distance_km / speed) * 60.0 + _INTERSECTION_DELAY


# ===========================================================================
# Traffic Graph — dynamic travel-time edge weights
# ===========================================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS-84 lat/lon points."""
    R  = 6371.0
    p1 = math.radians(lat1);  p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(max(a, 0.0)))


def _parse_coord(val) -> float:
    """Accept both '.' and ',' as decimal separators (SCATS CSV uses commas)."""
    return float(str(val).replace(",", "."))


class TrafficGraph:
    """
    Directed road network built from SCATS site metadata.

    Nodes  : unique SCATS numbers (int)
    Edges  : directed, stored as (to_id, distance_km)  — topology is fixed.
    Weights: computed on demand via calculate_travel_time(flow, dist_km).
    """

    def __init__(self):
        # node_id → (latitude, longitude)
        self.nodes: Dict[int, Tuple[float, float]] = {}
        # node_id → [(to_id, distance_km), ...]
        self.adj:   Dict[int, List[Tuple[int, float]]] = {}

    def add_node(self, node_id: int, lat: float, lon: float) -> None:
        self.nodes[node_id] = (lat, lon)
        self.adj.setdefault(node_id, [])

    def add_edge(self, from_id: int, to_id: int, distance_km: float) -> None:
        self.adj.setdefault(from_id, []).append((to_id, distance_km))

    def travel_time_adj(
        self,
        flow_map:       Dict[int, float],
        removed_edges:  Optional[Set[Tuple[int, int]]] = None,
        removed_nodes:  Optional[Set[int]]             = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Materialise a weighted adjacency dict with travel times as costs.

        Edges/nodes listed in *removed_edges* / *removed_nodes* are omitted —
        this is the temporary-removal mechanism required by Yen's spur step.
        """
        removed_edges = removed_edges or set()
        removed_nodes = removed_nodes or set()
        weighted: Dict[int, List[Tuple[int, float]]] = {}

        for from_id, neighbours in self.adj.items():
            if from_id in removed_nodes:
                continue
            flow = flow_map.get(from_id, 0.0)
            bucket: List[Tuple[int, float]] = []
            for to_id, dist_km in neighbours:
                if to_id in removed_nodes:
                    continue
                if (from_id, to_id) in removed_edges:
                    continue
                bucket.append((to_id, calculate_travel_time(flow, dist_km)))
            weighted[from_id] = bucket

        return weighted


def build_graph_from_scats(
    csv_path:       str,
    max_neighbours: int = 3,
) -> TrafficGraph:
    """
    Build a TrafficGraph from the SCATS CSV file.

    Each unique SCATS number becomes a node.  Directed edges connect every
    site to its `max_neighbours` nearest sites (haversine distance) in both
    directions so the graph is always navigable.
    """
    df = pd.read_csv(csv_path, sep=";", header=1)
    df["NB_LATITUDE"]  = df["NB_LATITUDE"].apply(_parse_coord)
    df["NB_LONGITUDE"] = df["NB_LONGITUDE"].apply(_parse_coord)

    sites = (
        df[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
        .drop_duplicates(subset="SCATS Number")
        .reset_index(drop=True)
    )

    graph = TrafficGraph()
    for _, row in sites.iterrows():
        graph.add_node(
            int(row["SCATS Number"]),
            float(row["NB_LATITUDE"]),
            float(row["NB_LONGITUDE"]),
        )

    node_ids = list(graph.nodes.keys())

    for nid in node_ids:
        lat1, lon1 = graph.nodes[nid]
        distances = [
            (_haversine(lat1, lon1, *graph.nodes[other]), other)
            for other in node_ids if other != nid
        ]
        distances.sort()
        seen_edges: Set[Tuple[int, int]] = set()
        for dist, other_id in distances[:max_neighbours]:
            for u, v in [(nid, other_id), (other_id, nid)]:
                if (u, v) not in seen_edges:
                    graph.add_edge(u, v, dist)
                    seen_edges.add((u, v))

    return graph


# ===========================================================================
# ML Model — per-site traffic flow prediction
# ===========================================================================

_MODEL_REGISTRY = {
    "lstm": lambda: TrafficLSTM(
        input_size=1, hidden_size=64, num_layers=2, dropout=0.2
    ),
    "gru": lambda: TrafficGRU(
        input_size=1, hidden_size=64, num_layers=2, dropout=0.2
    ),
    "transformer": lambda: TrafficTransformer(
        input_size=1, d_model=64, nhead=4,
        num_encoder_layers=2, dim_feedforward=128, dropout=0.1,
    ),
}

_PTH_NAMES = {
    "lstm":        "lstm_best.pth",
    "gru":         "gru_best.pth",
    "transformer": "transformer_best.pth",
}


def _load_trained_model(model_name: str) -> torch.nn.Module:
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_MODEL_REGISTRY)}"
        )
    pth = os.path.join(MODELS_DIR, _PTH_NAMES[key])
    if not os.path.exists(pth):
        raise FileNotFoundError(
            f"Saved weights not found: {pth}\n  → Run train.py first."
        )
    model = _MODEL_REGISTRY[key]()
    model.load_state_dict(torch.load(pth, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def predict_flows(
    model_name: str,
    time_step:  int = 12,
) -> Dict[int, float]:
    """
    Predict the next-step traffic flow (veh/hr) for every unique SCATS site.

    Strategy
    --------
    1. Fit a MinMaxScaler on the combined volume data from all sites and dates
       (identical to the TrafficDataProcessor training procedure).
    2. For each site's most-recent date, take the last *time_step* intervals
       as the input window, normalise with the shared scaler, run the model,
       and inverse-transform the prediction back to vehicle counts.

    Returns
    -------
    flow_map : {scats_number: predicted_flow_veh_per_hr}
    """
    volume_cols = [f"V{i:02d}" for i in range(96)]

    df = pd.read_csv(DATA_PATH, sep=";", header=1)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # --- Fit global scaler (mirrors TrafficDataProcessor._scale) -----------
    all_volumes: List[float] = []
    for _, row in df.iterrows():
        vals = pd.to_numeric(
            pd.Series(row[volume_cols].values, dtype=object), errors="coerce"
        )
        vals = vals.interpolate().ffill().bfill().fillna(0).tolist()
        all_volumes.extend(vals)

    global_scaler = MinMaxScaler(feature_range=(0, 1))
    global_scaler.fit(np.array(all_volumes, dtype=np.float32).reshape(-1, 1))

    # --- Load model --------------------------------------------------------
    nn_model = _load_trained_model(model_name)

    flow_map: Dict[int, float] = {}

    for scats_id, group in df.groupby("SCATS Number"):
        latest_row = group.sort_values("Date").iloc[-1]
        raw = pd.to_numeric(
            pd.Series(latest_row[volume_cols].values, dtype=object), errors="coerce"
        )
        raw = raw.interpolate().ffill().bfill().fillna(0).values.astype(np.float32)

        # Take the last time_step intervals as the prediction window
        window = raw[-time_step:].reshape(-1, 1)          # (time_step, 1)
        window_scaled = global_scaler.transform(window)    # (time_step, 1)

        # Shape: (1, time_step, 1)  →  batch=1, seq=time_step, features=1
        x = torch.tensor(
            window_scaled, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_scaled = nn_model(x).cpu().numpy()        # (1, 1)

        pred_flow = float(global_scaler.inverse_transform(pred_scaled)[0, 0])
        flow_map[int(scats_id)] = max(pred_flow, 0.0)

    return flow_map


# ===========================================================================
# REQUIREMENT 2 — Yen's K-Shortest Paths
# ===========================================================================

def _dijkstra(
    weighted_adj: Dict[int, List[Tuple[int, float]]],
    source:       int,
    target:       int,
) -> Tuple[Optional[List[int]], float]:
    """
    Dijkstra's shortest path on a pre-materialised weighted adjacency dict.

    Returns
    -------
    (path, cost)  or  (None, inf) if target is unreachable from source.
    """
    if source == target:
        return [source], 0.0

    dist = {source: 0.0}
    prev: Dict[int, int] = {}
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u == target:
            path: List[int] = []
            node = target
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(source)
            path.reverse()
            return path, d
        for v, w in weighted_adj.get(u, []):
            new_d = d + w
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(heap, (new_d, v))

    return None, math.inf


def yen_k_shortest_paths(
    graph:    TrafficGraph,
    flow_map: Dict[int, float],
    source:   int,
    target:   int,
    K:        int = 5,
) -> List[Tuple[List[int], float]]:
    """
    Yen's K-Shortest Loopless Paths algorithm.

    The edge weight for each link is ``calculate_travel_time(flow_at_source, dist_km)``,
    consistent with the PDF assumption that flow is taken from the starting SCATS site.

    Parameters
    ----------
    graph    : TrafficGraph (distance-based adjacency)
    flow_map : {node_id: predicted_flow_veh_hr}  — one entry per SCATS node
    source   : origin SCATS number
    target   : destination SCATS number
    K        : number of shortest paths to find

    Returns
    -------
    List of (path, total_travel_time_minutes), length ≤ K, sorted ascending.
    """
    # ---- Step 0: first shortest path ---------------------------------------
    full_adj = graph.travel_time_adj(flow_map)
    first_path, first_cost = _dijkstra(full_adj, source, target)
    if first_path is None:
        return []

    # A[k] = (cost, path) — confirmed k-th shortest paths
    A: List[Tuple[float, List[int]]] = [(first_cost, first_path)]

    # B = min-heap of candidate paths; tiebreaker counter avoids list comparison
    _tie = itertools.count()
    B: List[Tuple[float, int, List[int]]] = []

    # ---- Step 1 to K-1 -----------------------------------------------------
    for k in range(1, K):
        prev_path = A[k - 1][1]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            # Edges to remove: those leaving `spur_node` in every confirmed
            # path that shares `root_path` as a prefix.
            removed_edges: Set[Tuple[int, int]] = set()
            for _, confirmed in A:
                if (
                    len(confirmed) > i
                    and confirmed[: i + 1] == root_path
                ):
                    removed_edges.add((confirmed[i], confirmed[i + 1]))

            # Nodes to remove: every node in root_path except the spur node
            # itself (prevents loops back through the root).
            removed_nodes: Set[int] = set(root_path[:-1])

            # Dijkstra from spur_node to target on the pruned graph
            spur_adj = graph.travel_time_adj(
                flow_map, removed_edges, removed_nodes
            )
            spur_path, spur_cost = _dijkstra(spur_adj, spur_node, target)
            if spur_path is None:
                continue

            # Re-compute root-path cost from the full (un-pruned) adjacency
            root_cost = 0.0
            for j in range(len(root_path) - 1):
                u, v = root_path[j], root_path[j + 1]
                for nb, tt in full_adj.get(u, []):
                    if nb == v:
                        root_cost += tt
                        break

            # Candidate = root (minus duplicate spur_node) + spur path
            candidate_path = root_path[:-1] + spur_path
            candidate_cost = root_cost + spur_cost

            # Skip duplicates already in B
            if not any(p == candidate_path for _, _, p in B):
                heapq.heappush(B, (candidate_cost, next(_tie), candidate_path))

        if not B:
            break  # No more distinct paths exist

        best_cost, _, best_path = heapq.heappop(B)
        A.append((best_cost, best_path))

    return [(path, cost) for cost, path in A]


# ===========================================================================
# Public API
# ===========================================================================

def get_top_5_routes(
    origin:      int,
    destination: int,
    model_name:  str,
) -> List[Tuple[List[int], float]]:
    """
    Find the 5 shortest routes (by estimated travel time) between two SCATS
    sites using Yen's K-Shortest Paths algorithm with ML-predicted flows.

    Parameters
    ----------
    origin      : SCATS number of the starting intersection
    destination : SCATS number of the destination intersection
    model_name  : "lstm" | "gru" | "transformer"

    Returns
    -------
    routes : List[Tuple[List[int], float]]
        Up to 5 entries of (node_path, total_travel_time_minutes),
        ordered from shortest to longest.
    """
    print(
        f"\n[TBRGS] Route search: {origin} → {destination}  "
        f"(model: {model_name})"
    )

    # ---- Build graph -------------------------------------------------------
    graph = build_graph_from_scats(DATA_PATH)

    if origin not in graph.nodes:
        raise ValueError(
            f"Origin SCATS {origin} not found in graph. "
            f"Available nodes: {sorted(graph.nodes)}"
        )
    if destination not in graph.nodes:
        raise ValueError(
            f"Destination SCATS {destination} not found in graph. "
            f"Available nodes: {sorted(graph.nodes)}"
        )

    # ---- Predict flows -----------------------------------------------------
    print("  Predicting per-site traffic flows …")
    flow_map = predict_flows(model_name)
    for nid, flow in sorted(flow_map.items()):
        print(f"    SCATS {nid:>5d}  →  predicted flow = {flow:7.1f} veh/hr")

    # ---- Yen's K=5 ---------------------------------------------------------
    routes = yen_k_shortest_paths(
        graph, flow_map, origin, destination, K=5
    )

    if not routes:
        print("  ⚠  No path found between origin and destination.")
        return []

    print(f"\n  Top {len(routes)} route(s) by estimated travel time:")
    for rank, (path, tt) in enumerate(routes, 1):
        print(
            f"  {rank}. [{' → '.join(map(str, path))}]  "
            f"≈ {tt:.2f} min"
        )

    return routes


# ===========================================================================
# CLI smoke-test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("calculate_travel_time — sanity checks")
    print("=" * 60)
    test_cases = [
        (0,    1.0, "flow=0    → speed capped at 60"),
        (100,  1.0, "flow=100  → speed capped at 60"),
        (351,  1.0, "flow=351  → speed ~60 (boundary)"),
        (500,  1.0, "flow=500  → congested, speed < 60"),
        (1000, 1.0, "flow=1000 → heavily congested"),
        (1500, 1.0, "flow=1500 → capacity point, speed~32"),
    ]
    for flow, dist, label in test_cases:
        tt = calculate_travel_time(flow, dist)
        # Derive speed from time for display
        speed = dist / ((tt - _INTERSECTION_DELAY) / 60.0)
        print(f"  {label:<40s}  speed={speed:5.1f} km/h  tt={tt:.3f} min")

    print("\n" + "=" * 60)
    print("Graph construction from SCATS CSV")
    print("=" * 60)
    g = build_graph_from_scats(DATA_PATH)
    print(f"  Nodes : {sorted(g.nodes.keys())}")
    for nid, neighbours in g.adj.items():
        for nb, dist in neighbours:
            print(f"  Edge  : {nid} → {nb}  ({dist:.3f} km)")
