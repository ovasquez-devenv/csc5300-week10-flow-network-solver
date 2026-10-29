"""Min cost flow tests, including a check against an LP optimum."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow, total_cost_of_flow
from core.network import FlowNetwork
from utils.generators import random_digraph, random_sparse
from utils.validators import validate_flow


def lp_optimum(net: FlowNetwork, target_flow: float) -> float:
    """Solve the min cost flow as a linear program for an independent answer.

    The LP is the textbook formulation: flow on each edge stays within its
    capacity, flow is conserved at every node except the source and sink, and
    the total leaving the source equals target_flow. HiGHS solves it exactly.
    """
    edges = list(net.iter_edges())
    m = len(edges)
    costs = np.array([e.cost for e in edges], dtype=float)
    rows = []
    rhs = []
    for node in range(net.n):
        if node in (net.source, net.sink):
            continue
        row = np.zeros(m)
        for j, edge in enumerate(edges):
            if edge.v == node:
                row[j] += 1
            if edge.u == node:
                row[j] -= 1
        rows.append(row)
        rhs.append(0.0)
    supply = np.zeros(m)
    for j, edge in enumerate(edges):
        if edge.u == net.source:
            supply[j] = 1.0
    rows.append(supply)
    rhs.append(float(target_flow))

    result = linprog(
        costs,
        A_eq=np.array(rows),
        b_eq=np.array(rhs),
        bounds=[(0.0, e.capacity) for e in edges],
        method="highs",
    )
    assert result.success, result.message
    return float(result.fun)


def test_known_answer():
    net = FlowNetwork(4, 0, 3)
    net.add_edge(0, 1, 2, 1)
    net.add_edge(0, 2, 2, 2)
    net.add_edge(1, 3, 2, 3)
    net.add_edge(2, 3, 2, 1)
    result = min_cost_flow(net)
    assert result.value == pytest.approx(4.0)
    assert result.stats["cost"] == pytest.approx(14.0)


def test_respects_capacity_and_flow_value(tiny_network):
    net = tiny_network
    target = edmonds_karp(net.copy()).value
    result = min_cost_flow(net)
    assert result.value == pytest.approx(target)
    report = validate_flow(net, result.flow, 0, 5, result.value, claimed_cost=result.stats["cost"])
    assert report["ok"], report["problems"]


def test_costs_match_lp_optimum():
    """Independent optimum from a linear program, on six seeded instances."""
    for seed in (5300, 5301, 5302):
        for maker in (random_sparse,):
            net = maker(80, seed)
            target = edmonds_karp(net.copy()).value
            reference = lp_optimum(net, target)
            result = min_cost_flow(net.copy())
            assert result.stats["cost"] == pytest.approx(reference, abs=1e-6), (
                f"seed {seed}: solver {result.stats['cost']} vs LP {reference}"
            )


def test_limited_flow_is_cheaper_than_full_flow():
    """A smaller target should cost no more than the full flow and use it all."""
    net = random_sparse(60, 5300)
    full = min_cost_flow(net.copy())
    half = min_cost_flow(net.copy(), max_flow=full.value / 2.0)
    assert half.value == pytest.approx(full.value / 2.0, rel=1e-6)
    assert half.stats["cost"] <= full.stats["cost"] + 1e-6


def test_zero_target_returns_no_flow():
    net = random_sparse(30, 5300)
    result = min_cost_flow(net.copy(), max_flow=0.0)
    assert result.value == pytest.approx(0.0)
    assert result.stats["cost"] == pytest.approx(0.0)


def test_negative_costs_are_handled():
    """Negative costs on forward edges are legal as long as there is no cycle."""
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0, -2.0)
    net.add_edge(1, 2, 5.0, -1.0)
    result = min_cost_flow(net)
    assert result.value == pytest.approx(5.0)
    assert result.stats["cost"] == pytest.approx(-15.0)


def test_negative_cycle_is_rejected():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0, 1.0)
    net.add_edge(1, 2, 5.0, -4.0)
    net.add_edge(2, 1, 5.0, 1.0)
    with pytest.raises(ValueError):
        min_cost_flow(net)


def test_reverse_edge_cancels_bad_early_choice():
    """The classic case where a greedy first pick has to be undone.

    The cheapest first path is 0 to 1 to 3, but taking it blocks more flow than
    it gains. Sending flow and then cancelling it is what makes the answer
    optimal, and this test exists so a broken residual graph cannot hide.
    """
    net = FlowNetwork(4, 0, 3)
    net.add_edge(0, 1, 1, 1)
    net.add_edge(1, 2, 1, 1)
    net.add_edge(2, 3, 1, 1)
    net.add_edge(0, 2, 1, 5)
    net.add_edge(1, 3, 1, 5)
    result = min_cost_flow(net)
    assert result.value == pytest.approx(2.0)
    assert result.stats["cost"] == pytest.approx(12.0)


def test_path_trace_is_complete():
    net = random_sparse(40, 5300)
    result = min_cost_flow(net, trace=True)
    paths = result.stats["paths"]
    assert len(paths) == result.augmentations
    for path in paths:
        assert path[0][0] == net.source
        assert path[-1][1] == net.sink


def test_total_cost_helper():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 4.0, 3.0)
    net.add_edge(1, 2, 4.0, 2.0)
    result = min_cost_flow(net)
    assert total_cost_of_flow(net, result.flow) == pytest.approx(result.stats["cost"])
