"""Min cost max flow by successive shortest augmenting paths with potentials.

The idea is simple to state. Start with no flow. Repeatedly find the cheapest
path from source to sink in the residual graph, push as much as that path allows,
and stop when the sink is unreachable. Pushing along a residual reverse edge
cancels earlier flow, which is how the algorithm fixes a choice that looked
cheap early and turned out to block something better.

The catch is that residual edges carry negative costs, so a plain Dijkstra
cannot be used. Potentials fix that. A potential pi is a price per node, and the
reduced cost of an edge is cost + pi[u] - pi[v]. After each shortest path we
update pi[v] += dist[v], which keeps every reduced cost non negative and leaves
every shortest path the same length in reduced terms. This is Johnson's trick,
and it buys Dijkstra's O(E log V) instead of Bellman-Ford's O(V E) on every
round.

If the original graph has negative cost edges, the first potential comes from
one Bellman-Ford pass. A negative cost cycle in the original graph would make
the problem unbounded, so that case raises an error.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from core.network import FlowNetwork, FlowResult

INF = float("inf")


def _has_negative_cost_edges(net: FlowNetwork) -> bool:
    """True when any forward edge has a negative cost."""
    costs = net.costs
    for i in range(0, len(costs), 2):
        if costs[i] < 0:
            return True
    return False


def _initial_potentials(net: FlowNetwork, s: int) -> List[float]:
    """Shortest distances from s, used as the first price vector.

    When every forward cost is non negative, nothing beats zero, so the zero
    vector is already valid and this returns at once. That is the common case
    and it matters: Bellman-Ford over a 50,000 node graph with 400,000 edges is
    billions of relaxations, while the zero vector costs nothing.

    With negative costs one Bellman-Ford pass is required. Only nodes reachable
    from the source are considered, because an unreachable node has no bearing
    on any path from source to sink.
    """
    n = net.n
    if not _has_negative_cost_edges(net):
        return [0.0] * n

    heads = net.heads
    caps = net.capacities
    costs = net.costs
    dist = [INF] * n
    dist[s] = 0.0

    for _ in range(n - 1):
        changed = False
        for i in range(0, len(heads), 2):
            if caps[i] <= 0:
                continue
            u = heads[i ^ 1]
            v = heads[i]
            if dist[u] < INF and dist[u] + costs[i] < dist[v] - 1e-12:
                dist[v] = dist[u] + costs[i]
                changed = True
        if not changed:
            break
    for i in range(n):
        if dist[i] == INF:
            dist[i] = 0.0
    return dist


def _check_no_negative_cycle(net: FlowNetwork, s: int, potentials: List[float]) -> None:
    """One extra Bellman-Ford round on reachable nodes only.

    If a relaxation still succeeds, some reachable node can be improved after
    V - 1 rounds, which means a negative cost cycle exists and the cheapest flow
    would be unbounded. Nodes that the source cannot reach are skipped: they can
    never appear on a source to sink path, and treating them as zero distance
    is what made this check fire on ordinary graphs.
    """
    if not _has_negative_cost_edges(net):
        return

    n = net.n
    heads = net.heads
    caps = net.capacities
    costs = net.costs
    dist = list(potentials)
    for i in range(0, len(heads), 2):
        if caps[i] <= 0:
            continue
        u = heads[i ^ 1]
        v = heads[i]
        if dist[u] < INF and dist[u] + costs[i] < dist[v] - 1e-12:
            raise ValueError("negative cost cycle detected, min cost flow is unbounded")


def min_cost_flow(
    net: FlowNetwork,
    source: Optional[int] = None,
    sink: Optional[int] = None,
    max_flow: Optional[float] = None,
    trace: bool = False,
) -> FlowResult:
    """Push flow from source to sink at the lowest total cost.

    max_flow defaults to infinity, which means push as much as possible. Pass a
    smaller number to stop early, which is what the scheduling application uses
    when it only needs a target amount of flow.

    Returns a FlowResult whose value is the flow sent and whose stats["cost"] is
    the total cost of that flow. With max_flow set, the result is the cheapest
    flow of that value. With max_flow unset, it is the cheapest flow of the
    maximum value.
    """
    s = net.source if source is None else source
    t = net.sink if source is None else sink
    if s is None or t is None:
        raise ValueError("source and sink must be set")
    if s == t:
        raise ValueError("source and sink must differ")

    n = net.n
    heads = net.heads
    caps = net.capacities
    costs = net.costs
    adj = net.adjacency
    target = INF if max_flow is None else float(max_flow)

    pi = _initial_potentials(net, s)
    _check_no_negative_cycle(net, s, pi)

    total_flow = 0.0
    total_cost = 0.0
    rounds = 0
    path_costs: List[float] = []
    path_flows: List[float] = []
    paths: List[List[Tuple[int, int, float, float]]] = []

    dist = [INF] * n
    prev_edge = [-1] * n
    done = [False] * n
    touched: List[int] = []

    while total_flow < target - 1e-12:
        # Only the nodes the last search touched need clearing. On a 50,000
        # node graph that is the difference between a trivial cost and
        # millions of wasted assignments over the run.
        for i in touched:
            dist[i] = INF
            prev_edge[i] = -1
            done[i] = False
        touched.clear()

        dist[s] = 0.0
        touched.append(s)
        heap = [(0.0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            pi_u = pi[u]
            for i in adj[u]:
                cap = caps[i]
                if cap <= 0:
                    continue
                v = heads[i]
                # Reduced cost is never negative once potentials are valid.
                reduced = costs[i] + pi_u - pi[v]
                if reduced < 0:
                    reduced = 0.0
                nd = d + reduced
                if nd < dist[v] - 1e-12:
                    if dist[v] == INF:
                        touched.append(v)
                    dist[v] = nd
                    prev_edge[v] = i
                    heapq.heappush(heap, (nd, v))

        if dist[t] == INF:
            break

        # Shift the potentials by the real distances. Every node in touched was
        # settled by the search, so its distance is exact. Nodes that were never
        # reached have no residual edge into them from a settled node, so their
        # potentials can stay where they are without making any reduced cost
        # negative. An earlier version stopped the search at the sink and used
        # min(dist, dist[t]) for the rest, which produced valid looking flows
        # with a cost above the true optimum.
        for v in touched:
            pi[v] += dist[v]

        # Bottleneck along the found path.
        bottleneck = target - total_flow
        v = t
        real_cost = 0.0
        while v != s:
            i = prev_edge[v]
            if caps[i] < bottleneck:
                bottleneck = caps[i]
            real_cost += costs[i]
            v = heads[i ^ 1]
        if bottleneck <= 0:
            break

        if trace:
            path: List[Tuple[int, int, float, float]] = []
            v = t
            while v != s:
                i = prev_edge[v]
                path.append((heads[i ^ 1], v, caps[i], costs[i]))
                v = heads[i ^ 1]
            path.reverse()
            paths.append(path)

        v = t
        while v != s:
            i = prev_edge[v]
            caps[i] -= bottleneck
            caps[i ^ 1] += bottleneck
            v = heads[i ^ 1]

        total_flow += bottleneck
        total_cost += bottleneck * real_cost
        rounds += 1
        path_costs.append(real_cost)
        path_flows.append(bottleneck)

    result = FlowResult(
        value=total_flow,
        flow=net.flow_map(),
        algorithm="min_cost_flow",
        augmentations=rounds,
        iterations=rounds,
    )
    result.stats = {
        "cost": total_cost,
        "path_costs": path_costs,
        "path_flows": path_flows,
        "mean_path_cost": (sum(path_costs) / len(path_costs)) if path_costs else 0.0,
    }
    if trace:
        result.stats["paths"] = paths
    return result


def min_cost_for_flow(
    net: FlowNetwork,
    source: Optional[int] = None,
    sink: Optional[int] = None,
    flow_value: float = 0.0,
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """Cheapest flow of exactly flow_value units. Returns (cost, flow map)."""
    result = min_cost_flow(net, source, sink, max_flow=flow_value)
    if result.value < flow_value - 1e-9:
        raise ValueError(
            f"cannot send {flow_value} units, the network only allows {result.value}"
        )
    return result.stats["cost"], result.flow


def total_cost_of_flow(net: FlowNetwork, flow: Dict[Tuple[int, int], float]) -> float:
    """Cost of a given sparse flow, using the original edge costs."""
    cost = 0.0
    for (u, v), f in flow.items():
        for edge in net.iter_edges():
            if edge.u == u and edge.v == v:
                cost += f * edge.cost
                break
    return cost
