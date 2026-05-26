"""
test_system.py
==============
Pytest test suite for the Traffic-Based Route Guidance System (TBRGS).

Coverage (15 tests)
-------------------
 Group A — Travel-time maths          (tests 01-05)
 Group B — Graph & pathfinding        (tests 06-10)
 Group C — Edge / boundary cases      (tests 11-13)
 Group D — Data pipeline              (test  14)
 Group E — Model weight fallback      (test  15)

Run
---
    pytest test_system.py -v
"""

import math
import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from the Workspace directory
# ---------------------------------------------------------------------------
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

from routing import (
    calculate_travel_time,
    TrafficGraph,
    yen_k_shortest_paths,
    _FLOW_THRESHOLD,
    _SPEED_LIMIT,
    _INTERSECTION_DELAY,
    _A,
    _B,
)

# ── Constants used in assertions ──────────────────────────────────────────
_TOL = 1e-6   # floating-point tolerance


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_graph():
    """
    A small, fully-connected 5-node TrafficGraph built from scratch.

    Topology (directed, bidirectional):

        1 ──(1.0)── 2 ──(1.0)── 3
        │                         │
       (2.0)                    (1.5)
        │                         │
        4 ──────(1.5)────────── 5

    Node 1 → Node 3 has three distinct simple paths (ranks 1-3).
    """
    g = TrafficGraph()
    nodes = [(1, -37.80, 145.00),
             (2, -37.81, 145.01),
             (3, -37.82, 145.02),
             (4, -37.83, 145.00),
             (5, -37.82, 145.01)]
    for nid, lat, lon in nodes:
        g.add_node(nid, lat, lon)

    edges = [
        (1, 2, 1.0), (2, 1, 1.0),
        (2, 3, 1.0), (3, 2, 1.0),
        (1, 4, 2.0), (4, 1, 2.0),
        (4, 5, 1.5), (5, 4, 1.5),
        (5, 3, 1.5), (3, 5, 1.5),
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d)

    return g


@pytest.fixture
def uniform_flow(simple_graph):
    """Flow map with 200 veh/hr for every node → speed capped at 60 km/h."""
    return {nid: 200.0 for nid in simple_graph.nodes}


@pytest.fixture
def congested_flow(simple_graph):
    """Flow map with 800 veh/hr for every node → congested, speed < 60 km/h."""
    return {nid: 800.0 for nid in simple_graph.nodes}


# ===========================================================================
# Group A — Travel-Time Mathematics (tests 01-05)
# ===========================================================================

class TestCalculateTravelTime:
    """Verify the flow-to-speed conversion formula and travel-time output."""

    def test_01_flow_below_threshold_caps_speed_at_60(self):
        """
        flow=200 ≤ 351 → speed must be capped at 60 km/h.
        Expected: (1.0 / 60) * 60 + 0.5 = 1.5 min exactly.
        """
        tt = calculate_travel_time(200, 1.0)
        assert abs(tt - 1.5) < _TOL, f"Expected 1.5, got {tt}"

    def test_02_flow_at_threshold_still_uses_speed_limit(self):
        """
        flow=351 is at the boundary and must still be capped at 60 km/h.
        Expected travel time identical to flow=200 for same distance.
        """
        tt_at_threshold = calculate_travel_time(351, 1.0)
        tt_well_below   = calculate_travel_time(200, 1.0)
        assert abs(tt_at_threshold - tt_well_below) < _TOL

    def test_03_flow_above_threshold_increases_travel_time(self):
        """
        flow=800 > 351 → quadratic branch → speed < 60 → tt > 1.5 min.
        Expected value: ≈ 1.6140 min (regression anchor).
        """
        tt = calculate_travel_time(800, 1.0)
        expected = 1.6139900588
        assert abs(tt - expected) < 1e-5, f"Expected ≈{expected}, got {tt}"

    def test_04_congested_flow_gives_lower_speed_than_free_flow(self):
        """
        Higher flow → lower speed → higher travel time (monotone property).
        """
        tt_free      = calculate_travel_time(200,  1.0)
        tt_moderate  = calculate_travel_time(800,  1.0)
        tt_congested = calculate_travel_time(1200, 1.0)
        assert tt_free < tt_moderate < tt_congested

    def test_05_travel_time_scales_linearly_with_distance(self):
        """
        For a fixed flow in the capped region, doubling distance should
        double the link component (not the intersection delay):
            tt(flow, 2d) = 2 * link_component(flow, d) + delay
                         = tt(flow, d) - delay + tt(flow, d)  ... not simply 2×tt
        Exact check: tt(200, 2.0) = 2.5 min.
        """
        tt = calculate_travel_time(200, 2.0)
        assert abs(tt - 2.5) < _TOL, f"Expected 2.5, got {tt}"

    def test_05b_quadratic_round_trip(self):
        """
        Solve speed from flow=800, plug back into the parabola, and verify
        we recover the original flow within floating-point tolerance.
        """
        flow = 800.0
        tt   = calculate_travel_time(flow, 1.0)
        speed = 1.0 / ((tt - _INTERSECTION_DELAY) / 60.0)
        recovered = _A * speed ** 2 + _B * speed
        assert abs(recovered - flow) < 0.01, (
            f"Round-trip mismatch: predicted={flow}, recovered={recovered:.4f}"
        )


# ===========================================================================
# Group B — Graph Construction & Yen's Pathfinding (tests 06-10)
# ===========================================================================

class TestYenKShortestPaths:
    """Verify Yen's algorithm on the simple_graph fixture."""

    def test_06_returns_up_to_5_distinct_paths(self, simple_graph, uniform_flow):
        """
        Yen's algorithm must return ≤ 5 paths, all of which are distinct.
        """
        routes = yen_k_shortest_paths(simple_graph, uniform_flow, 1, 3, K=5)
        assert 1 <= len(routes) <= 5, f"Expected 1-5 routes, got {len(routes)}"
        paths = [tuple(p) for p, _ in routes]
        assert len(paths) == len(set(paths)), "Duplicate paths returned"

    def test_07_first_path_has_minimum_travel_time(self, simple_graph, uniform_flow):
        """
        The first returned path must be the one with the smallest travel time
        (routes are sorted ascending by travel time).
        """
        routes = yen_k_shortest_paths(simple_graph, uniform_flow, 1, 3, K=5)
        times  = [tt for _, tt in routes]
        assert times == sorted(times), "Routes are not sorted by travel time"

    def test_08_all_paths_connect_origin_to_destination(self, simple_graph, uniform_flow):
        """
        Every returned path must start at origin and end at destination.
        """
        origin, dest = 1, 3
        routes = yen_k_shortest_paths(simple_graph, uniform_flow, origin, dest, K=5)
        for path, _ in routes:
            assert path[0]  == origin, f"Path does not start at origin: {path}"
            assert path[-1] == dest,   f"Path does not end at destination: {path}"

    def test_09_congested_flow_increases_total_travel_time(
        self, simple_graph, uniform_flow, congested_flow
    ):
        """
        Under congested flow (800 veh/hr) the travel time on the best route
        must be strictly greater than under free-flow (200 veh/hr).
        """
        routes_free = yen_k_shortest_paths(
            simple_graph, uniform_flow, 1, 3, K=5
        )
        routes_cong = yen_k_shortest_paths(
            simple_graph, congested_flow, 1, 3, K=5
        )
        best_free = routes_free[0][1]
        best_cong = routes_cong[0][1]
        assert best_cong > best_free, (
            f"Congested time ({best_cong:.4f}) should exceed "
            f"free-flow time ({best_free:.4f})"
        )

    def test_10_travel_time_adj_excludes_removed_edges(self, simple_graph, uniform_flow):
        """
        TrafficGraph.travel_time_adj must honour the removed_edges set:
        requesting removal of (1,2) must make that edge absent from the
        returned adjacency dict.
        """
        removed = {(1, 2)}
        adj = simple_graph.travel_time_adj(uniform_flow, removed_edges=removed)
        neighbours_of_1 = [v for v, _ in adj.get(1, [])]
        assert 2 not in neighbours_of_1, (
            "Edge (1,2) should have been excluded but was still present"
        )


# ===========================================================================
# Group C — Edge Cases & Invalid Inputs (tests 11-13)
# ===========================================================================

class TestEdgeCases:

    def test_11_origin_equals_destination_returns_single_node_path(self, simple_graph, uniform_flow):
        """
        When origin == destination Yen's inner Dijkstra returns the trivial
        zero-cost path [node].  The outer Yen's loop should still produce a
        valid (possibly length-1) result without raising an exception.
        """
        routes = yen_k_shortest_paths(simple_graph, uniform_flow, 3, 3, K=5)
        # Must not raise; if routes are returned they should start and end at 3
        for path, _ in routes:
            assert path[0] == 3 and path[-1] == 3

    def test_12_no_path_returns_empty_list(self, simple_graph, uniform_flow):
        """
        If there is no path between two nodes (isolated node 99 added),
        yen_k_shortest_paths must return an empty list rather than raising.
        """
        simple_graph.add_node(99, -38.0, 146.0)   # isolated — no edges
        routes = yen_k_shortest_paths(simple_graph, uniform_flow, 1, 99, K=5)
        assert routes == [], f"Expected [], got {routes}"

    def test_13_invalid_flow_value_does_not_crash(self):
        """
        Negative or extremely large flows are clamped / handled gracefully
        by calculate_travel_time without raising an exception.
        """
        for flow in [-999, 0, 100_000]:
            tt = calculate_travel_time(flow, 1.0)
            assert isinstance(tt, float), f"Expected float, got {type(tt)} for flow={flow}"
            assert tt >= _INTERSECTION_DELAY, (
                f"Travel time {tt} must be ≥ intersection delay {_INTERSECTION_DELAY}"
            )


# ===========================================================================
# Group D — Data Pipeline (test 14)
# ===========================================================================

class TestDataPipeline:

    def test_14_dataloader_shapes_are_correct_for_rnn(self):
        """
        TrafficDataProcessor.build_loaders() must return DataLoaders whose
        batches have shape (batch, time_step, 1) for X and (batch, 1) for y,
        with all values in [0, 1] after MinMax scaling.
        """
        import torch
        from data_processing import TrafficDataProcessor

        DATA_CSV = os.path.join(WORKSPACE, "data", "scats_sample.csv")
        if not os.path.exists(DATA_CSV):
            pytest.skip("scats_sample.csv not found — skipping data pipeline test")

        TIME_STEP  = 12
        BATCH_SIZE = 32

        processor = TrafficDataProcessor(
            filepath=DATA_CSV,
            time_step=TIME_STEP,
            batch_size=BATCH_SIZE,
        )
        train_loader, val_loader, test_loader = processor.build_loaders()

        X_batch, y_batch = next(iter(train_loader))

        # Shape
        assert X_batch.ndim == 3, (
            f"X must be 3-D (batch, seq, features), got ndim={X_batch.ndim}"
        )
        assert X_batch.shape[1] == TIME_STEP, (
            f"Sequence length must equal time_step={TIME_STEP}, "
            f"got {X_batch.shape[1]}"
        )
        assert X_batch.shape[2] == 1, (
            f"Feature dimension must be 1, got {X_batch.shape[2]}"
        )
        assert y_batch.ndim == 2 and y_batch.shape[1] == 1, (
            f"y must be shape (batch, 1), got {tuple(y_batch.shape)}"
        )

        # Values in [0, 1]
        assert float(X_batch.min()) >= -1e-6, "X contains values below 0"
        assert float(X_batch.max()) <= 1.0 + 1e-6, "X contains values above 1"


# ===========================================================================
# Group E — Missing Model Weights Fallback (test 15)
# ===========================================================================

class TestModelFallback:

    def test_15_missing_pth_raises_file_not_found(self, tmp_path, monkeypatch):
        """
        _load_trained_model should raise FileNotFoundError (not silently fail)
        when the .pth weights file does not exist.  The routing worker in the
        GUI catches this and falls back to mock flows, so an explicit raise
        here is the correct contract.
        """
        from routing import _load_trained_model

        # Redirect MODELS_DIR to an empty temporary directory
        import routing as _routing_module
        monkeypatch.setattr(_routing_module, "MODELS_DIR", str(tmp_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_trained_model("lstm")

        assert "lstm_best.pth" in str(exc_info.value), (
            "FileNotFoundError message should name the missing .pth file"
        )
