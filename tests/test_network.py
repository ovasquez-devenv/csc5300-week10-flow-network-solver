"""Tests for the network container itself."""

from __future__ import annotations

import pytest

from core.network import FlowNetwork


def test_add_edge_creates_reverse_edge():
    net = FlowNetwork(2)
    index = net.add_edge(0, 1, 5.0, 2.0)
    assert net.edge_count == 2
    assert net.original_edge_count == 1
    # Forward carries the capacity, reverse starts at zero.
    assert net.capacities[index] == 5.0
    assert net.capacities[index + 1] == 0.0
    # Costs are opposite, which is what makes cancelling flow cost neutral.
    assert net.costs[index] == 2.0
    assert net.costs[index + 1] == -2.0


def test_reverse_edge_is_paired_by_xor():
    net = FlowNetwork(3)
    first = net.add_edge(0, 1, 4.0)
    second = net.add_edge(1, 2, 3.0)
    assert second == first + 2
    assert (first ^ 1) == first + 1


def test_add_edge_rejects_self_loop():
    net = FlowNetwork(3)
    with pytest.raises(ValueError):
        net.add_edge(1, 1, 2.0)


def test_add_edge_rejects_bad_endpoint():
    net = FlowNetwork(3)
    with pytest.raises(IndexError):
        net.add_edge(0, 5, 1.0)


def test_add_edge_rejects_negative_capacity():
    net = FlowNetwork(2)
    with pytest.raises(ValueError):
        net.add_edge(0, 1, -1.0)


def test_flow_map_reflects_used_capacity(tiny_network):
    net = tiny_network
    net.capacities[0] -= 7.0
    net.capacities[1] += 7.0
    flow = net.flow_map()
    assert flow[(0, 1)] == pytest.approx(7.0)


def test_parallel_edges_are_summed(tiny_network):
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 2.0)
    net.add_edge(0, 1, 3.0)
    net.add_edge(1, 2, 5.0)
    # Spend both parallel edges fully.
    for i in (0, 2):
        net.capacities[i] = 0.0
        net.capacities[i + 1] = net._orig_cap[i]
    flow = net.flow_map()
    assert flow[(0, 1)] == pytest.approx(5.0)


def test_reset_restores_capacities(tiny_network):
    net = tiny_network
    original = list(net.capacities)
    net.capacities[0] = 0.0
    net.reset()
    assert net.capacities == original


def test_copy_is_independent(tiny_network):
    net = tiny_network
    clone = net.copy()
    clone.capacities[0] = 0.0
    assert net.capacities[0] != 0.0
    clone.add_edge(0, 5, 1.0)
    assert net.edge_count != clone.edge_count


def test_flow_balance_of_simple_flow():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0)
    net.add_edge(1, 2, 5.0)
    balance = net.flow_balance({(0, 1): 3.0, (1, 2): 3.0})
    assert balance[0] == pytest.approx(3.0)
    assert balance[1] == pytest.approx(0.0)
    assert balance[2] == pytest.approx(-3.0)


def test_conservation_error_spots_bad_flow():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0)
    net.add_edge(1, 2, 5.0)
    # One unit vanishes at node 1.
    error = net.conservation_error({(0, 1): 4.0, (1, 2): 3.0}, 0, 2)
    assert error == pytest.approx(1.0)


def test_capacity_violations_reports_overflow():
    net = FlowNetwork(2, 0, 1)
    net.add_edge(0, 1, 3.0)
    bad = net.capacity_violations({(0, 1): 4.0})
    assert len(bad) == 1
    u, v, f, cap = bad[0]
    assert (u, v) == (0, 1)
    assert f == pytest.approx(4.0)
    assert cap == pytest.approx(3.0)


def test_residual_graph_lists_positive_edges(tiny_network):
    net = tiny_network
    n, edges = net.get_residual_graph()
    assert n == net.n
    # Only forward edges carry positive residual capacity before any flow runs,
    # because every reverse edge starts at zero.
    assert len(edges) == net.original_edge_count
    assert all(cap > 0 for _, _, cap in edges)


def test_min_cut_partitions_all_nodes(tiny_network):
    from core.edmonds_karp import edmonds_karp

    net = tiny_network
    result = edmonds_karp(net)
    left, right, cut = net.min_cut(0, result.flow)
    assert left | right == set(range(net.n))
    assert not (left & right)
    assert cut == pytest.approx(23.0)


def test_iter_edges_reports_flow(tiny_network):
    from core.edmonds_karp import edmonds_karp

    net = tiny_network
    edmonds_karp(net)
    for edge in net.iter_edges():
        assert 0.0 <= edge.flow <= edge.capacity + 1e-9


def test_add_node_extends_network():
    net = FlowNetwork(1)
    node = net.add_node("extra")
    assert node == 1
    assert net.n == 2
    assert net.name_of(node) == "extra"
