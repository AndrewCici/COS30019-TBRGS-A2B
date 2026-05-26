"""
test_system.py
==============
Pytest test suite for the Traffic-Based Route Guidance System (TBRGS).
Exactly 15 test cases.

Coverage
--------
  Group A — Travel-time maths          tests 01-05
  Group B — Graph & Yen's pathfinding  tests 06-10
  Group C — Edge / boundary cases      tests 11-13
  Group D — Data pipeline              test  14
  Group E — Model weight fallback      test  15

Run
---
    pytest test_system.py -v
"""

import math
import os
import sys

import pytest

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

from routing import (
    calculate_travel_time,
    TrafficGraph,
    yen_k_shortest_paths,
    _A,
    _B,
    _FLOW_THRESHOLD,
    _SPEED_LIMIT,
    _INTERSECTION_DELAY,
)

_TOL = 1e-6


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_graph():
    """4-node graph used for basic Yen's property tests (≥ 2 simple paths)."""
    g = TrafficGraph()
    for nid, lat, lon in [(1, -37.80, 145.00), (2, -37.81, 145.01),
                           (3, -37.82, 145.02), (4, -37.83, 145.00)]:
        g.add_node(nid, lat, lon)
    for u, v, d in [(1, 2, 1.0), (2, 1, 1.0),
                    (2, 3, 1.0), (3, 2, 1.0),
                    (1, 4, 2.0), (4, 1, 2.0),
                    (4, 3, 1.5), (3, 4, 1.5)]:
        g.add_edge(u, v, d)
    return g


@pytest.fixture
def rich_graph():
    """
    6-node DAG-like graph with exactly 8 simple paths from node 1 to node 6,
    guaranteeing Yen's K=5 returns exactly 5 routes.

    Topology:
        1 → 2 → 4 → 6
        1 → 2 → 4 → 5 → 6
        1 → 2 → 5 → 6
        1 → 2 → 5 → 4 → 6
        1 → 3 → 4 → 6
        1 → 3 → 4 → 5 → 6
        1 → 3 → 5 → 6
        1 → 3 → 5 → 4 → 6
    """
    g = TrafficGraph()
    for nid, lat, lon in [(1, -37.80, 145.00), (2, -37.81, 145.01),
                           (3, -37.81, 145.00), (4, -37.82, 145.01),
                           (5, -37.82, 145.00), (6, -37.83, 145.01)]:
        g.add_node(nid, lat, lon)
    for u, v, d in [(1, 2, 1.0), (1, 3, 1.2),
                    (2, 4, 1.0), (2, 5, 1.5),
                    (3, 4, 1.2), (3, 5, 1.0),
                    (4, 6, 1.0), (4, 5, 0.8),
                    (5, 6, 1.0), (5, 4, 0.8)]:
        g.add_edge(u, v, d)
    return g


@pytest.fixture
def free_flow(rich_graph):
    return {nid: 200.0 for nid in rich_graph.nodes}


@pytest.fixture
def congested_flow(rich_graph):
    return {nid: 800.0 for nid in rich_graph.nodes}


# ===========================================================================
# Group A — Travel-Time Mathematics (tests 01-05)
# ===========================================================================

def test_01_flow_below_threshold_caps_speed_at_60_kmh():
    """
    flow=200 ≤ 351 → speed capped at 60 km/h.
    Expected: (1.0/60)*60 + 0.5 = 1.5 min exactly.
    """
    tt = calculate_travel_time(200, 1.0)
    assert abs(tt - 1.5) < _TOL, f"Expected 1.5, got {tt}"


def test_02_flow_at_boundary_351_still_uses_speed_limit():
    """
    flow=351 is the exact threshold — must still be capped (≤, not <).
    """
    tt_boundary = calculate_travel_time(351, 1.0)
    tt_below    = calculate_travel_time(200, 1.0)
    assert abs(tt_boundary - tt_below) < _TOL, (
        f"Boundary flow=351 should match capped result {tt_below}, got {tt_boundary}"
    )


def test_03_flow_above_threshold_uses_quadratic_green_branch():
    """
    flow=800 > 351 → quadratic solved for higher (under-capacity) root.
    Regression anchor: tt ≈ 1.6140 min for distance=1 km.
    """
    tt = calculate_travel_time(800, 1.0)
    assert abs(tt - 1.6139900588) < 1e-5, f"Expected ≈1.6140, got {tt}"


def test_04_travel_time_increases_monotonically_with_flow():
    """
    The model must be monotone: higher flow → lower speed → higher travel time.
    Tested across the capped region, boundary, and quadratic region.
    """
    flows = [0, 200, 351, 500, 800, 1200]
    times = [calculate_travel_time(f, 1.0) for f in flows]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Monotone violated: tt({flows[i]}) = {times[i]} < tt({flows[i-1]}) = {times[i-1]}"
        )


def test_05_quadratic_round_trip_recovers_original_flow():
    """
    Solve speed from flow=800, substitute back into the parabola, and verify
    the recovered flow matches within 0.01 veh/hr — confirms the correct
    (green / higher) root is selected.
    """
    target_flow = 800.0
    tt    = calculate_travel_time(target_flow, 1.0)
    speed = 1.0 / ((tt - _INTERSECTION_DELAY) / 60.0)
    recovered = _A * speed ** 2 + _B * speed
    assert abs(recovered - target_flow) < 0.01, (
        f"Round-trip mismatch: input={target_flow}, recovered={recovered:.4f}"
    )


# ===========================================================================
# Group B — Graph Construction & Yen's K-Shortest Paths (tests 06-10)
# ===========================================================================

def test_06_yen_returns_exactly_5_routes(rich_graph, free_flow):
    """
    With K=5 and ≥ 8 simple paths available, Yen's must return exactly 5.
    """
    routes = yen_k_shortest_paths(rich_graph, free_flow, 1, 6, K=5)
    assert len(routes) == 5, f"Expected exactly 5 routes, got {len(routes)}"


def test_07_routes_are_sorted_ascending_by_travel_time(rich_graph, free_flow):
    """
    Routes must be ordered from shortest to longest travel time.
    """
    routes = yen_k_shortest_paths(rich_graph, free_flow, 1, 6, K=5)
    times  = [tt for _, tt in routes]
    assert times == sorted(times), f"Routes not sorted: {times}"


def test_08_every_path_starts_at_origin_and_ends_at_destination(rich_graph, free_flow):
    """
    Every returned path must start at the origin node and end at the
    destination node.
    """
    origin, dest = 1, 6
    routes = yen_k_shortest_paths(rich_graph, free_flow, origin, dest, K=5)
    for path, _ in routes:
        assert path[0]  == origin, f"Path does not start at origin: {path}"
        assert path[-1] == dest,   f"Path does not end at destination: {path}"


def test_09_congested_flow_strictly_increases_best_route_time(
    rich_graph, free_flow, congested_flow
):
    """
    Best-route travel time under congested flow (800 veh/hr) must be strictly
    greater than under free-flow (200 veh/hr) on the same graph.
    """
    best_free = yen_k_shortest_paths(rich_graph, free_flow,    1, 6, K=1)[0][1]
    best_cong = yen_k_shortest_paths(rich_graph, congested_flow, 1, 6, K=1)[0][1]
    assert best_cong > best_free, (
        f"Congested time ({best_cong:.4f}) must exceed free-flow ({best_free:.4f})"
    )


def test_10_travel_time_adj_honours_removed_edges(simple_graph):
    """
    TrafficGraph.travel_time_adj must omit edges listed in removed_edges.
    """
    flow_map = {nid: 200.0 for nid in simple_graph.nodes}
    adj      = simple_graph.travel_time_adj(flow_map, removed_edges={(1, 2)})
    neighbours_of_1 = [v for v, _ in adj.get(1, [])]
    assert 2 not in neighbours_of_1, (
        "Edge (1→2) should have been excluded but is still present in adj"
    )


# ===========================================================================
# Group C — Edge Cases & Invalid Inputs (tests 11-13)
# ===========================================================================

def test_11_origin_equals_destination_does_not_raise(simple_graph):
    """
    When origin == destination the function must not raise and must return a
    valid (trivial) path that starts and ends at the same node.
    """
    flow_map = {nid: 200.0 for nid in simple_graph.nodes}
    routes   = yen_k_shortest_paths(simple_graph, flow_map, 3, 3, K=5)
    for path, _ in routes:
        assert path[0] == path[-1] == 3


def test_12_unreachable_destination_returns_empty_list(simple_graph):
    """
    Adding an isolated node (no edges) and routing to it must return [] rather
    than raising an exception.
    """
    simple_graph.add_node(99, -39.0, 147.0)   # isolated
    flow_map = {nid: 200.0 for nid in simple_graph.nodes}
    flow_map[99] = 200.0
    routes = yen_k_shortest_paths(simple_graph, flow_map, 1, 99, K=5)
    assert routes == [], f"Expected [] for unreachable node, got {routes}"


def test_13_extreme_and_invalid_flow_values_do_not_crash():
    """
    calculate_travel_time must handle negative, zero, and very large flow
    values without raising — and always return a value ≥ intersection delay.
    """
    for flow in [-9999, -1, 0, 1, 100_000, 1_000_000]:
        tt = calculate_travel_time(flow, 1.0)
        assert isinstance(tt, float), f"Expected float, got {type(tt)} for flow={flow}"
        assert tt >= _INTERSECTION_DELAY, (
            f"Travel time {tt} < intersection delay {_INTERSECTION_DELAY} for flow={flow}"
        )


# ===========================================================================
# Group D — Data Pipeline (test 14)
# ===========================================================================

def test_14_dataloader_batch_shape_matches_rnn_requirements():
    """
    TrafficDataProcessor.build_loaders() must produce batches shaped
    (batch_size, time_step, 1) for X and (batch_size, 1) for y, with all
    X values in [0, 1] after MinMax scaling.
    """
    import torch
    from data_processing import TrafficDataProcessor

    csv = os.path.join(WORKSPACE, "data", "scats_clean.csv")
    if not os.path.exists(csv):
        pytest.skip("scats_clean.csv not found")

    TIME_STEP  = 12
    BATCH_SIZE = 32

    proc = TrafficDataProcessor(csv, time_step=TIME_STEP, batch_size=BATCH_SIZE)
    train_loader, _, _ = proc.build_loaders()

    X, y = next(iter(train_loader))

    assert X.ndim == 3,                 f"X must be 3-D, got ndim={X.ndim}"
    assert X.shape[1] == TIME_STEP,     f"Seq length must be {TIME_STEP}, got {X.shape[1]}"
    assert X.shape[2] == 1,             f"Feature dim must be 1, got {X.shape[2]}"
    assert y.ndim == 2 and y.shape[1] == 1, f"y must be (batch,1), got {tuple(y.shape)}"
    assert float(X.min()) >= -1e-6,     "X contains values below 0 after scaling"
    assert float(X.max()) <= 1.0 + 1e-6, "X contains values above 1 after scaling"


# ===========================================================================
# Group E — Missing Model Weights Fallback (test 15)
# ===========================================================================

def test_15_missing_pth_raises_file_not_found_with_filename(tmp_path, monkeypatch):
    """
    _load_trained_model must raise FileNotFoundError when the .pth file is
    absent, and the error message must name the missing file.  This is the
    contract the GUI RoutingWorker catches to fall back to mock flows.
    """
    import routing as _r
    monkeypatch.setattr(_r, "MODELS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        _r._load_trained_model("lstm")

    assert "lstm_best.pth" in str(exc_info.value), (
        f"Error message should name 'lstm_best.pth', got: {exc_info.value}"
    )
