"""Generator tests: same seed same graph, different seed different graph."""

from __future__ import annotations

import numpy as np
import pytest

from utils.generators import (
    bipartite_matching_graph,
    capacity_summary,
    grid_graph,
    layered_network,
    random_costs,
    random_dense,
    random_digraph,
    random_sparse,
)


def _fingerprint(net):
    """Everything that defines a network, in a comparable form."""
    return (
        net.n,
        net.source,
        net.sink,
        tuple((e.u, e.v, e.capacity, e.cost) for e in net.iter_edges()),
    )


def test_sparse_is_reproducible():
    assert _fingerprint(random_sparse(200, 5300)) == _fingerprint(random_sparse(200, 5300))


def test_dense_is_reproducible():
    assert _fingerprint(random_dense(200, 5300)) == _fingerprint(random_dense(200, 5300))


def test_different_seed_gives_different_graph():
    a = _fingerprint(random_sparse(200, 5300))
    b = _fingerprint(random_sparse(200, 5301))
    assert a != b


def test_digraph_is_reproducible():
    assert _fingerprint(random_digraph(40, 0.2, 5302)) == _fingerprint(random_digraph(40, 0.2, 5302))


def test_grid_is_reproducible():
    assert _fingerprint(grid_graph(10, 10, 5303)) == _fingerprint(grid_graph(10, 10, 5303))


def test_bipartite_is_reproducible():
    assert bipartite_matching_graph(10, 10, 3, 5304) == bipartite_matching_graph(10, 10, 3, 5304)


def test_sparse_has_the_requested_shape():
    net = random_sparse(500, 5300, fan_out=2)
    assert 450 <= net.n <= 550
    summary = capacity_summary(net)
    assert summary["edges"] > 0
    assert summary["max_capacity"] <= 100.0
    assert summary["mean_capacity"] > 0


def test_dense_has_more_edges_than_sparse():
    sparse = capacity_summary(random_sparse(800, 5300))
    dense = capacity_summary(random_dense(800, 5300))
    assert dense["edges"] > sparse["edges"]


def test_layered_network_has_source_and_sink():
    net = layered_network(4, 5, 2, 5300)
    assert net.source == 0
    assert net.sink == net.n - 1
    # Every layer must be reachable from the source and lead to the sink.
    reachable = net.reachable_from(net.source)
    assert net.sink in reachable


def test_bipartite_edges_are_in_range():
    left, right, edges = bipartite_matching_graph(12, 8, 3, 5300)
    assert all(0 <= u < left and 0 <= v < right for u, v in edges)
    assert len(edges) == 12 * 3


def test_grid_node_and_edge_counts():
    net = grid_graph(8, 9, 5300)
    assert net.n == 8 * 9 + 2
    # Horizontal edges, vertical edges, both directions, plus source and sink.
    horizontal = 8 * (9 - 1) * 2
    vertical = (8 - 1) * 9 * 2
    unary = 8 * 9 * 2
    assert net.original_edge_count == horizontal + vertical + unary


def test_capacity_distributions_all_work():
    for distribution in ("uniform", "exponential", "power_law"):
        net = random_sparse(120, 5300, distribution=distribution)
        caps = [e.capacity for e in net.iter_edges()]
        assert caps
        assert min(caps) >= 1.0
        assert all(c >= 0 for c in caps)


def test_unknown_distribution_raises():
    with pytest.raises(ValueError):
        random_sparse(60, 5300, distribution="cauchy")


def test_random_costs_are_symmetric():
    net = random_sparse(60, 5300)
    random_costs(net, 5300)
    costs = net.costs
    for i in range(0, len(costs), 2):
        assert costs[i + 1] == -costs[i]


def test_bad_layer_count_raises():
    with pytest.raises(ValueError):
        layered_network(1, 5, 2, 5300)
