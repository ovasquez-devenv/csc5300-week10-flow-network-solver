"""Bipartite matching by reduction to max flow.

The reduction is the standard one. Add a source that points at every left node
with capacity 1, point every left node at the right nodes it is allowed to match
with capacity 1, and point every right node at a sink with capacity 1. An
integral max flow then picks out a set of disjoint pairs, and the max flow value
is the size of the largest matching. Capacity 1 on every edge is what forces
each node to be used at most once.

Max flow with integer capacities always has an integer optimum, and both solvers
here push whole units along edges with whole capacities, so the flow that comes
out is already integral. No rounding step is needed.

Weighted matching uses the same structure with costs, solved by min cost flow.
That gives the assignment problem: maximize total weight instead of just the
count of pairs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import FlowNetwork, FlowResult


def build_matching_network(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
    weights: Optional[Dict[Tuple[int, int], float]] = None,
) -> Tuple[FlowNetwork, int, int]:
    """Build the flow network for a bipartite matching instance.

    Left nodes are 1..n_left, right nodes are n_left + 1..n_left + n_right,
    the source is 0 and the sink is the last node. weights, if given, become the
    negative costs on the left to right edges so that min cost flow maximizes
    total weight. Returns (network, source, sink).
    """
    source = 0
    left_base = 1
    right_base = 1 + n_left
    sink = right_base + n_right
    net = FlowNetwork(sink + 1, source, sink)

    for u in range(n_left):
        net.add_edge(source, left_base + u, 1.0)
    for v in range(n_right):
        net.add_edge(right_base + v, sink, 1.0)
    for u, v in edges:
        if not (0 <= u < n_left and 0 <= v < n_right):
            raise IndexError(f"edge ({u}, {v}) outside the bipartite graph")
        cost = 0.0
        if weights is not None:
            # Cheapest flow maximizes weight, so negate.
            cost = -float(weights.get((u, v), 0.0))
        net.add_edge(left_base + u, right_base + v, 1.0, cost)
    return net, source, sink


def bipartite_matching(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
    solver: str = "ek",
) -> List[Tuple[int, int]]:
    """Maximum cardinality matching as a sorted list of (left, right) pairs."""
    net, s, t = build_matching_network(n_left, n_right, edges)
    if solver == "ek":
        edmonds_karp(net, s, t)
    elif solver == "pr":
        from core.push_relabel import push_relabel

        push_relabel(net, s, t)
    else:
        raise ValueError(f"unknown solver: {solver}")

    left_base = 1
    right_base = 1 + n_left
    pairs: List[Tuple[int, int]] = []
    for (u, v), f in net.flow_map().items():
        if f <= 0:
            continue
        if left_base <= u < left_base + n_left and right_base <= v < right_base + n_right:
            pairs.append((u - left_base, v - right_base))
    pairs.sort()
    return pairs


def maximum_weight_matching(
    n_left: int,
    n_right: int,
    edges: Sequence[Tuple[int, int]],
    weights: Dict[Tuple[int, int], float],
    solver_result: bool = False,
):
    """Matching of maximum total weight, solved with min cost max flow.

    Returns the pairs and the total weight, or the full FlowResult if
    solver_result is true.
    """
    net, s, t = build_matching_network(n_left, n_right, edges, weights)
    result = min_cost_flow(net, s, t)

    left_base = 1
    right_base = 1 + n_left
    pairs: List[Tuple[int, int]] = []
    total_weight = 0.0
    for (u, v), f in result.flow.items():
        if f <= 0:
            continue
        if left_base <= u < left_base + n_left and right_base <= v < right_base + n_right:
            lu = u - left_base
            rv = v - right_base
            pairs.append((lu, rv))
            total_weight += weights.get((lu, rv), 0.0)
    pairs.sort()
    if solver_result:
        result.stats["weight"] = total_weight
        return result
    return pairs, total_weight


def is_valid_matching(pairs: Sequence[Tuple[int, int]], n_left: int, n_right: int) -> bool:
    """True when no node appears twice and every index is in range."""
    lefts = [u for u, _ in pairs]
    rights = [v for _, v in pairs]
    if len(set(lefts)) != len(lefts) or len(set(rights)) != len(rights):
        return False
    return all(0 <= u < n_left for u in lefts) and all(0 <= v < n_right for v in rights)


def matching_to_matrix(pairs: Sequence[Tuple[int, int]], n_left: int, n_right: int):
    """0/1 numpy matrix with a 1 wherever a pair was chosen."""
    import numpy as np

    mat = np.zeros((n_left, n_right), dtype=int)
    for u, v in pairs:
        mat[u, v] = 1
    return mat


def brute_force_max_matching(n_left: int, n_right: int, edges: Sequence[Tuple[int, int]]) -> int:
    """Exponential reference answer for small instances, used by the tests."""
    allowed = set(edges)
    best = 0

    def search(u: int, used: set, count: int) -> None:
        nonlocal best
        if u == n_left:
            best = max(best, count)
            return
        # Option 1: leave this left node unmatched.
        search(u + 1, used, count)
        # Option 2: match it with any free right node it allows.
        for v in range(n_right):
            if (u, v) in allowed and v not in used:
                used.add(v)
                search(u + 1, used, count + 1)
                used.remove(v)

    search(0, set(), 0)
    return best


def summarize_matching(pairs: Sequence[Tuple[int, int]], labels_left=None, labels_right=None) -> str:
    """One line per pair, using names when they are supplied."""
    lines = []
    for u, v in pairs:
        left = labels_left[u] if labels_left else str(u)
        right = labels_right[v] if labels_right else str(v)
        lines.append(f"  {left} -> {right}")
    return "\n".join(lines)
