"""Push-relabel max flow with the highest label selection rule.

Push-relabel works locally instead of searching for a path from source to sink.
Every node holds an excess (flow that arrived but has not left) and a height.
A node may push along an edge when its height is exactly one above the other
end, where "downhill" means toward the sink. If no edge is downhill, the node
is relabeled to one above its lowest neighbor with residual capacity.

Variants implemented here:

- highest_label (default): always discharge the active node with the largest
  height. With the gap heuristic and global relabeling this is the fastest of
  the three variants in practice on the graphs we build.
- fifo: discharge active nodes in the order they became active.

Everything is loop based and uses flat lists indexed by node id. There is no
recursion, so a 50,000 node graph is no problem for the call stack.

Reference: CLRS Chapter 26, section 26.4, with the discharge procedure from
Goldberg and Tarjan (1988).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from core.network import Deadline, FlowNetwork, FlowResult, TimeoutError_

INF = float("inf")


class _State:
    """Working arrays for one push-relabel run."""

    __slots__ = (
        "excess", "height", "cur", "count", "buckets", "active", "in_queue",
        "max_height", "queue", "ops", "pushes", "relabels", "saturating_pushes",
        "gaps", "global_relabels", "profile", "trace", "next_sample", "use_queue",
        "relabels_since_global", "relabel_threshold",
    )

    def __init__(self, n: int):
        self.excess = [0.0] * n
        self.height = [0] * n
        self.cur = [0] * n
        self.count = [0] * (2 * n + 4)
        self.buckets: List[List[int]] = [[] for _ in range(2 * n + 4)]
        self.active = [False] * n
        self.in_queue = [False] * n
        self.max_height = 0
        self.queue = deque()
        self.ops = 0
        self.pushes = 0
        self.relabels = 0
        self.saturating_pushes = 0
        self.gaps = 0
        self.global_relabels = 0
        self.profile: List[Tuple[int, float]] = []
        self.trace: List[Tuple] = []
        self.next_sample = 1
        self.use_queue = False
        self.relabels_since_global = 0
        self.relabel_threshold = 10 ** 9


def _global_relabel(net: FlowNetwork, s: int, t: int, st: _State) -> None:
    """Set exact distance to the sink for every node that can reach it.

    The distances are computed by a BFS backwards over residual edges, which is
    O(V + E) and resets all the heights at once. Nodes that cannot reach the
    sink get height n + 1 so they stop pushing.
    """
    n = net.n
    heads = net.heads
    caps = net.capacities
    adj = net.adjacency

    height = st.height
    for i in range(n):
        height[i] = n + 1
    height[t] = 0
    queue = deque([t])
    while queue:
        v = queue.popleft()
        h = height[v] + 1
        for i in adj[v]:
            # Edge i points v -> heads[i]. The edge back is i ^ 1, so a residual
            # edge heads[i] -> v exists when cap[i ^ 1] is positive.
            if caps[i ^ 1] > 0:
                u = heads[i]
                if height[u] == n + 1:
                    height[u] = h
                    queue.append(u)
    # The source keeps its own height so it can always push out of its edges.
    height[s] = n

    count = st.count
    for i in range(len(count)):
        count[i] = 0
    for i in range(n):
        count[height[i]] += 1
    for i in range(n):
        st.cur[i] = 0
        st.buckets[i] = []
    for u in range(n):
        if u != s and u != t and st.excess[u] > 0 and height[u] <= n:
            st.buckets[height[u]].append(u)
            st.active[u] = True
    st.max_height = n
    st.global_relabels += 1


def _discharge(net: FlowNetwork, u: int, s: int, t: int, st: _State, sample_every_events: int) -> None:
    """Push all of u's excess downhill, relabeling when no edge is downhill."""
    heads = net.heads
    caps = net.capacities
    adj = net.adjacency
    excess = st.excess
    height = st.height
    cur = st.cur
    count = st.count
    n = net.n
    trace = st.trace
    record = len(trace) < 4000

    while excess[u] > 0:
        if cur[u] >= len(adj[u]):
            # Relabel: one above the lowest neighbor that still has capacity.
            current = height[u]
            lowest = 2 * n + 4
            for i in adj[u]:
                if caps[i] > 0:
                    hv = height[heads[i]]
                    if hv < lowest:
                        lowest = hv
            new_height = lowest + 1
            old = current
            height[u] = new_height
            count[old] -= 1
            if new_height < len(count):
                count[new_height] += 1
            cur[u] = 0
            st.relabels += 1
            st.relabels_since_global += 1
            st.ops += 1

            if record:
                trace.append(("relabel", u, old, new_height, excess[u]))

            # Gap heuristic. If no node is left at the old height, nothing can
            # ever reach the sink from above it, so raise the whole band.
            if count[old] == 0 and old < n:
                st.gaps += 1
                for v in range(n):
                    if v != s and v != t and old < height[v] < n:
                        count[height[v]] -= 1
                        height[v] = n + 1
                        count[n + 1] += 1
                        st.active[v] = False
                if record:
                    trace.append(("gap", old, 0, 0, 0.0))

            if new_height > st.max_height and new_height <= n:
                st.max_height = new_height
            if new_height > n:
                # Unreachable from the sink now. Its excess is dead weight and
                # can never return to the source, which is fine for max flow.
                return

            # Keep sampling so the convergence profile stays cheap on big graphs.
            if st.ops >= st.next_sample:
                st.profile.append((st.ops, excess[t]))
                if sample_every_events > 0:
                    st.next_sample = st.ops + sample_every_events
                else:
                    st.next_sample = max(st.ops + 1, int(st.ops * 1.2))
            continue

        i = adj[u]
        idx = cur[u]
        edge = i[idx]
        v = heads[edge]
        if caps[edge] > 0 and height[u] == height[v] + 1:
            remaining = caps[edge]
            amount = excess[u] if excess[u] < remaining else remaining
            caps[edge] = remaining - amount
            caps[edge ^ 1] += amount
            excess[u] -= amount
            excess[v] += amount
            st.pushes += 1
            st.ops += 1
            if amount == remaining:
                st.saturating_pushes += 1
            if record:
                trace.append(("push", u, v, amount, excess[v]))
            if v != s and v != t and not st.active[v] and excess[v] > 0 and height[v] <= n:
                st.active[v] = True
                st.buckets[height[v]].append(v)
                if st.use_queue:
                    # In FIFO mode a node that just became active has to be
                    # queued, or its excess would never be discharged.
                    st.queue.append(v)
                if height[v] > st.max_height:
                    st.max_height = height[v]
            if st.ops >= st.next_sample:
                st.profile.append((st.ops, excess[t]))
                if sample_every_events > 0:
                    st.next_sample = st.ops + sample_every_events
                else:
                    st.next_sample = max(st.ops + 1, int(st.ops * 1.2))
        else:
            cur[u] += 1


def _return_excess(net: FlowNetwork, s: int, t: int, st: _State) -> int:
    """Send leftover excess back to the source so the result is a real flow.

    Push-relabel stops with a valid preflow: flow conservation holds everywhere
    except that some nodes may still hold excess. Those nodes are exactly the
    ones that could not reach the sink. Every unit of that excess arrived along
    some path from the source, so the reverse of that path still has residual
    capacity.

    The method here is a backward breadth first search from the source over
    residual edges, which builds a tree of return routes, followed by one pass
    that moves excess up the tree deepest node first. Deepest first matters: a
    node hands its excess to its parent before the parent hands anything on, so
    a unit moves up only once per pass.

    A single tree may not be enough. A node's excess may have arrived along one
    set of edges while the tree hands it back along another whose residual
    capacity is already spent, so the pass leaves something behind. The tree is
    therefore rebuilt and the pass repeated until nothing moves, which takes a
    handful of rounds in practice. The result is a true flow whose value equals
    the preflow value at the sink, which is the max flow.

    Returns the number of excess carrying moves made.
    """
    n = net.n
    heads = net.heads
    caps = net.capacities
    adj = net.adjacency
    excess = st.excess
    moved_total = 0

    for _ in range(n + 2):
        parent_edge = [-1] * n
        parent_node = [-1] * n
        depth = [-1] * n
        depth[s] = 0
        queue = deque([s])
        order = [s]
        while queue:
            v = queue.popleft()
            for i in adj[v]:
                # Edge i runs v -> heads[i]. Walking back from v to heads[i]
                # needs residual capacity on the edge the other way, i ^ 1.
                if caps[i ^ 1] > 0:
                    u = heads[i]
                    if depth[u] < 0 and u != t:
                        # Returning excess through the sink would undo good flow.
                        depth[u] = depth[v] + 1
                        # Edge i runs v -> u, so pushing u's excess to v means
                        # using the paired edge i ^ 1. Keep both the edge and
                        # the node, because heads[i] is u itself, not v.
                        parent_edge[u] = i ^ 1
                        parent_node[u] = v
                        queue.append(u)
                        order.append(u)

        moved = 0
        for start in reversed(order):
            if start == s or start == t or excess[start] <= 1e-12:
                continue
            i = parent_edge[start]
            if i < 0:
                continue
            home = parent_node[start]
            room = caps[i]
            amount = excess[start] if excess[start] < room else room
            if amount <= 0:
                continue
            caps[i] -= amount
            caps[i ^ 1] += amount
            excess[start] -= amount
            excess[home] += amount
            moved += 1
            moved_total += 1

        if moved == 0:
            break
        if all(excess[u] <= 1e-12 for u in range(n) if u != s and u != t):
            break
    return moved_total


def push_relabel(
    net: FlowNetwork,
    source: Optional[int] = None,
    sink: Optional[int] = None,
    variant: str = "highest_label",
    trace: bool = False,
    profile_points: int = 300,
    deadline: Optional[Deadline] = None,
) -> FlowResult:
    """Run push-relabel on net, updating net's residual capacities in place.

    variant is "highest_label" or "fifo". trace stores push, relabel, and gap
    events for the animation and is meant for small graphs. profile_points
    controls roughly how many samples of (operations, flow at sink) are kept for
    the convergence plots.
    """
    s = net.source if source is None else source
    t = net.sink if source is None else sink
    if s is None or t is None:
        raise ValueError("source and sink must be set")
    if s == t:
        raise ValueError("source and sink must differ")
    if variant not in ("highest_label", "fifo"):
        raise ValueError(f"unknown variant: {variant}")

    n = net.n
    heads = net.heads
    caps = net.capacities
    adj = net.adjacency
    st = _State(n)
    st.relabel_threshold = max(64, 4 * (net.original_edge_count ** 0.5) * 8)
    st.use_queue = variant == "fifo"
    st.height[s] = n
    st.excess = [0.0] * n
    st.count[n] = n - 1 if n > 1 else 1

    # Sample budget: dense enough to draw a curve, sparse enough to be free.
    total_edges = max(1, len(heads))
    sample_every = 0 if total_edges < 2000 else max(1, total_edges // profile_points)

    # Saturate every edge out of the source. This is the standard start and it
    # leaves the source with no excess of its own.
    for i in adj[s]:
        if deadline is not None:
            deadline.tick()
        cap = caps[i]
        if cap > 0:
            v = heads[i]
            caps[i] = 0.0
            caps[i ^ 1] += cap
            st.excess[v] += cap
            st.excess[s] -= cap
            st.pushes += 1
            st.saturating_pushes += 1
            if trace:
                st.trace.append(("push", s, v, cap, st.excess[v]))
    st.ops += len(adj[s])

    st.cur = [0] * n
    _global_relabel(net, s, t, st)

    if variant == "fifo":
        for u in range(n):
            if u != s and u != t and st.excess[u] > 0 and st.height[u] <= n:
                st.queue.append(u)
                st.active[u] = True
                st.in_queue[u] = True
        while st.queue:
            u = st.queue.popleft()
            st.in_queue[u] = False
            st.active[u] = False
            if st.excess[u] <= 0 or st.height[u] > n:
                continue
            _discharge(net, u, s, t, st, sample_every)
            if st.excess[u] > 0 and st.height[u] <= n and not st.in_queue[u]:
                st.queue.append(u)
                st.in_queue[u] = True
                st.active[u] = True
    else:
        # Highest label: scan down from the largest occupied height.
        while st.max_height >= 0:
            bucket = st.buckets[st.max_height] if st.max_height < len(st.buckets) else []
            if not bucket:
                st.max_height -= 1
                continue
            u = bucket.pop()
            if not st.active[u] or st.excess[u] <= 0 or st.height[u] != st.max_height:
                continue
            if deadline is not None:
                deadline.tick()
            st.active[u] = False
            _discharge(net, u, s, t, st, sample_every)
            if st.excess[u] > 0 and st.height[u] <= n:
                st.active[u] = True
                st.buckets[st.height[u]].append(u)
                if st.height[u] > st.max_height:
                    st.max_height = st.height[u]

            # Periodic global relabel. Local relabels guess at the distance to
            # the sink one node at a time, and on a large graph those guesses
            # drift far from the truth, which is what produces millions of
            # pointless pushes. A fresh BFS every so often resets every height
            # to its exact value and collapses the work. The threshold grows
            # with the graph so small instances skip it entirely.
            if st.relabels_since_global >= st.relabel_threshold:
                _global_relabel(net, s, t, st)
                st.relabels_since_global = 0
                st.max_height = n
                bucket_limit = len(st.buckets) - 1
                while st.max_height > 0 and not st.buckets[st.max_height]:
                    st.max_height -= 1
                if st.max_height > bucket_limit:
                    st.max_height = bucket_limit

    st.profile.append((st.ops, st.excess[t]))
    return_paths = _return_excess(net, s, t, st)

    result = FlowResult(
        value=st.excess[t],
        flow=net.flow_map(),
        algorithm=f"push_relabel_{variant}",
        augmentations=st.pushes,
        iterations=st.relabels,
    )
    result.stats = {
        "variant": variant,
        "pushes": st.pushes,
        "relabels": st.relabels,
        "saturating_pushes": st.saturating_pushes,
        "gaps": st.gaps,
        "global_relabels": st.global_relabels,
        "operations": st.ops,
        "profile": st.profile,
        "max_height": st.max_height,
        "excess_return_paths": return_paths,
    }
    if trace:
        result.stats["trace"] = st.trace
    return result


def max_flow_value(net: FlowNetwork, source: Optional[int] = None, sink: Optional[int] = None) -> float:
    return push_relabel(net, source, sink).value
