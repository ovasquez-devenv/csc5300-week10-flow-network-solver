"""Edmonds-Karp tests, including the max flow min cut identity on random graphs."""

from __future__ import annotations

import pytest

from core.edmonds_karp import edmonds_karp, max_flow_value
from utils.generators import random_digraph
from utils.validators import validate_flow


def test_known_answer(tiny_network):
    result = edmonds_karp(tiny_network)
    assert result.value == pytest.approx(23.0)


def test_returns_valid_flow(tiny_network):
    net = tiny_network
    result = edmonds_karp(net)
    report = validate_flow(net, result.flow, 0, 5, result.value)
    assert report["ok"], report["problems"]


def test_flow_equals_min_cut_on_random_graphs(seeds):
    """The max flow min cut theorem, checked on five seeded graphs.

    The cut is recomputed from the flow by the reachable set method, and its
    capacity is compared with the flow value. They must be equal for every one.
    """
    for seed in seeds:
        net = random_digraph(25, 0.25, seed)
        result = edmonds_karp(net.copy())
        left, right, cut = net.min_cut(0, result.flow)
        assert cut == pytest.approx(result.value, abs=1e-6), f"seed {seed}"


def test_no_augmenting_path_remains(tiny_network):
    net = tiny_network
    result = edmonds_karp(net)
    residual = net.copy()
    residual.reset()
    caps = residual.capacities
    heads = residual.heads
    for i in range(0, len(heads), 2):
        u = heads[i ^ 1]
        v = heads[i]
        used = result.flow.get((u, v), 0.0)
        caps[i] = max(0.0, caps[i] - used)
        caps[i ^ 1] = caps[i ^ 1] + used
    reachable = residual.reachable_from(0)
    assert 5 not in reachable


def test_augmentation_bookkeeping(tiny_network):
    result = edmonds_karp(tiny_network)
    assert result.augmentations == len(result.stats["bottlenecks"])
    assert result.augmentations == len(result.stats["path_lengths"])
    assert result.iterations == result.augmentations + 1
    assert all(b > 0 for b in result.stats["bottlenecks"])


def test_paths_are_recorded_on_request(tiny_network):
    result = edmonds_karp(tiny_network, keep_paths=True)
    paths = result.stats["paths"]
    assert len(paths) == result.augmentations
    for path in paths:
        assert path[0][0] == 0
        assert path[-1][1] == 5
        # Consecutive edges must join up.
        for (u1, v1, _), (u2, v2, _) in zip(path, path[1:]):
            assert v1 == u2


def test_disconnected_sink_returns_zero():
    from core.network import FlowNetwork

    net = FlowNetwork(4, 0, 3)
    net.add_edge(0, 1, 5.0)
    net.add_edge(2, 3, 5.0)
    result = edmonds_karp(net)
    assert result.value == 0.0
    assert result.augmentations == 0


def test_zero_capacity_edges_are_ignored():
    from core.network import FlowNetwork

    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 0.0)
    net.add_edge(1, 2, 5.0)
    assert edmonds_karp(net).value == 0.0


def test_single_edge_carries_its_capacity():
    from core.network import FlowNetwork

    net = FlowNetwork(2, 0, 1)
    net.add_edge(0, 1, 7.5)
    result = edmonds_karp(net)
    assert result.value == pytest.approx(7.5)
    assert result.flow[(0, 1)] == pytest.approx(7.5)


def test_convenience_wrapper_matches(tiny_network):
    assert max_flow_value(tiny_network.copy()) == pytest.approx(23.0)


def test_missing_source_raises():
    from core.network import FlowNetwork

    net = FlowNetwork(3)
    net.add_edge(0, 1, 1.0)
    with pytest.raises(ValueError):
        edmonds_karp(net)


def test_same_source_and_sink_raises(tiny_network):
    with pytest.raises(ValueError):
        edmonds_karp(tiny_network, 2, 2)
