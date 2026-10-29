"""Bipartite matching tests: validity, maximality, and the weighted version."""

from __future__ import annotations

import pytest

from applications.matching import (
    bipartite_matching,
    brute_force_max_matching,
    build_matching_network,
    is_valid_matching,
    matching_to_matrix,
    maximum_weight_matching,
)
from utils.generators import bipartite_matching_graph


def test_valid_matching_on_known_instance():
    edges = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    pairs = bipartite_matching(3, 3, edges)
    assert is_valid_matching(pairs, 3, 3)
    assert len(pairs) == 3


def test_result_is_maximum_for_small_cases(seeds):
    """Compare against brute force on five seeded instances."""
    for seed in seeds:
        left, right, edges = bipartite_matching_graph(6, 6, 2, seed)
        pairs = bipartite_matching(left, right, edges)
        expected = brute_force_max_matching(left, right, edges)
        assert len(pairs) == expected, f"seed {seed}: got {len(pairs)}, best is {expected}"


def test_result_is_valid_for_larger_cases(seeds):
    for seed in seeds:
        left, right, edges = bipartite_matching_graph(30, 25, 4, seed)
        pairs = bipartite_matching(left, right, edges)
        assert is_valid_matching(pairs, left, right)
        # Every pair must be one of the allowed edges.
        allowed = set(edges)
        assert all(pair in allowed for pair in pairs)


def test_unbalanced_sides():
    """More left nodes than right nodes caps the matching at the right count."""
    edges = [(0, 0), (1, 0), (2, 0), (3, 1)]
    pairs = bipartite_matching(4, 2, edges)
    assert len(pairs) == 2
    assert is_valid_matching(pairs, 4, 2)


def test_no_edges_gives_empty_matching():
    assert bipartite_matching(5, 5, []) == []


def test_push_relabel_solver_matches_edmonds_karp():
    left, right, edges = bipartite_matching_graph(20, 20, 3, 5300)
    ek = bipartite_matching(left, right, edges, solver="ek")
    pr = bipartite_matching(left, right, edges, solver="pr")
    assert len(ek) == len(pr)


def test_matrix_has_one_per_pair():
    pairs = [(0, 1), (2, 0)]
    mat = matching_to_matrix(pairs, 3, 3)
    assert mat.sum() == 2
    assert mat[0, 1] == 1
    assert mat[2, 0] == 1


def test_invalid_matching_is_rejected():
    assert not is_valid_matching([(0, 0), (0, 1)], 3, 3)
    assert not is_valid_matching([(0, 0), (1, 0)], 3, 3)
    assert not is_valid_matching([(5, 0)], 3, 3)


def test_edge_outside_graph_raises():
    with pytest.raises(IndexError):
        build_matching_network(3, 3, [(0, 7)])


def test_weighted_matching_prefers_high_weights():
    edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
    weights = {(0, 0): 1, (0, 1): 9, (1, 0): 9, (1, 1): 1}
    pairs, total = maximum_weight_matching(2, 2, edges, weights)
    assert total == pytest.approx(18.0)
    assert set(pairs) == {(0, 1), (1, 0)}


def test_weighted_matching_matches_brute_force():
    """Enumerate every matching on a tiny instance and compare the best weight."""
    import itertools

    left, right, edges = bipartite_matching_graph(4, 4, 2, 5300)
    weights = {(u, v): (u * 7 + v * 3) % 11 + 1 for u, v in edges}
    pairs, total = maximum_weight_matching(left, right, edges, weights)

    best = 0.0
    for size in range(1, min(left, right) + 1):
        for combo in itertools.combinations(edges, size):
            if len({u for u, _ in combo}) != size or len({v for _, v in combo}) != size:
                continue
            best = max(best, sum(weights[p] for p in combo))
    assert total == pytest.approx(best)


def test_weighted_result_is_valid_matching():
    left, right, edges = bipartite_matching_graph(10, 10, 3, 5302)
    weights = {(u, v): (u + v) % 5 + 1 for u, v in edges}
    pairs, _ = maximum_weight_matching(left, right, edges, weights)
    assert is_valid_matching(pairs, left, right)
