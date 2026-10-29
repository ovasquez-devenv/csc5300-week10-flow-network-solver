"""Edmonds-Karp max flow.

Edmonds-Karp is Ford-Fulkerson with a breadth first search for the shortest
augmenting path in the residual graph, counted in edges. Shortest paths keep
the number of augmentations at most O(V E), which is what turns an algorithm
that can loop forever on bad capacities into a polynomial one.

The search is iterative on purpose. Recursion would blow the stack on the
50,000 node graphs the assignment asks for.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from core.network import Deadline, FlowNetwork, FlowResult, TimeoutError_

INF = float("inf")


def _bfs_path(net: FlowNetwork, s: int, t: int, parent_edge: List[int], seen_stamp: List[int], stamp: int) -> bool:
    """Fill parent_edge with the shortest augmenting path. True if t was reached.

    seen_stamp holds the last search number that touched a node, which avoids
    clearing a 50,000 entry list on every BFS.
    """
    heads = net.heads
    caps = net.capacities
    adj = net.adjacency

    queue = deque([s])
    seen_stamp[s] = stamp
    parent_edge[s] = -1
    while queue:
        u = queue.popleft()
        for i in adj[u]:
            if caps[i] > 0:
                v = heads[i]
                if seen_stamp[v] != stamp:
                    seen_stamp[v] = stamp
                    parent_edge[v] = i
                    if v == t:
                        return True
                    queue.append(v)
    return False


def edmonds_karp(
    net: FlowNetwork,
    source: Optional[int] = None,
    sink: Optional[int] = None,
    keep_paths: bool = False,
    max_augmentations: Optional[int] = None,
    deadline: Optional[Deadline] = None,
) -> FlowResult:
    """Run Edmonds-Karp on net, updating net's residual capacities in place.

    source and sink default to net.source and net.sink. keep_paths stores every
    augmenting path, which is only sensible for small graphs and is what the
    animation uses. max_augmentations and deadline are safety valves for
    benchmarks: hitting either raises TimeoutError_ rather than returning a
    partial flow that looks like an answer.
    """
    s = net.source if source is None else source
    t = net.sink if sink is None else sink
    if s is None or t is None:
        raise ValueError("source and sink must be set")
    if s == t:
        raise ValueError("source and sink must differ")

    n = net.n
    heads = net.heads
    caps = net.capacities
    parent_edge = [-1] * n
    seen_stamp = [-1] * n
    stamp = 0

    total = 0.0
    augmentations = 0
    bfs_rounds = 0
    bottlenecks: List[float] = []
    path_lengths: List[int] = []
    paths: List[List[Tuple[int, int, float]]] = []

    while True:
        if deadline is not None:
            deadline.tick()
        stamp += 1
        found = _bfs_path(net, s, t, parent_edge, seen_stamp, stamp)
        bfs_rounds += 1
        if not found:
            break

        # Walk the path backwards to find the bottleneck, then forwards to push.
        bottleneck = INF
        v = t
        length = 0
        while v != s:
            i = parent_edge[v]
            if caps[i] < bottleneck:
                bottleneck = caps[i]
            v = heads[i ^ 1]
            length += 1

        if keep_paths:
            path: List[Tuple[int, int, float]] = []
            v = t
            while v != s:
                i = parent_edge[v]
                path.append((heads[i ^ 1], v, caps[i]))
                v = heads[i ^ 1]
            path.reverse()
            paths.append(path)

        v = t
        while v != s:
            i = parent_edge[v]
            caps[i] -= bottleneck
            caps[i ^ 1] += bottleneck
            v = heads[i ^ 1]

        total += bottleneck
        augmentations += 1
        bottlenecks.append(bottleneck)
        path_lengths.append(length)

        if max_augmentations is not None and augmentations >= max_augmentations:
            raise TimeoutError_(f"stopped after {max_augmentations} augmentations")

    result = FlowResult(
        value=total,
        flow=net.flow_map(),
        algorithm="edmonds_karp",
        augmentations=augmentations,
        iterations=bfs_rounds,
    )
    result.stats = {
        "bottlenecks": bottlenecks,
        "path_lengths": path_lengths,
        "mean_path_length": (sum(path_lengths) / len(path_lengths)) if path_lengths else 0.0,
        "max_path_length": max(path_lengths) if path_lengths else 0,
        "bfs_rounds": bfs_rounds,
    }
    if keep_paths:
        result.stats["paths"] = paths
    return result


def max_flow_value(net: FlowNetwork, source: Optional[int] = None, sink: Optional[int] = None) -> float:
    """Convenience wrapper when only the value is needed."""
    return edmonds_karp(net, source, sink).value


def min_cut_value(net: FlowNetwork, source: Optional[int] = None, sink: Optional[int] = None) -> Tuple[float, set, set]:
    """Max flow and the matching min cut, via max flow min cut."""
    result = edmonds_karp(net, source, sink)
    s = net.source if source is None else source
    left, right, cut = net.min_cut(s, result.flow)
    return result.value, left, right
