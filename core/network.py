"""Core flow network container shared by every solver in this project.

The network stores residual capacities in flat Python lists. Edge i and its
paired reverse edge i ^ 1 always sit next to each other, so the reverse of an
edge is found with a single xor. Storing the residual capacity instead of a
separate flow array keeps the solvers small and makes validation easy.

Node ids are integers from 0 to n - 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

INF = float("inf")


class TimeoutError_(Exception):
    """Raised when a solver hits the time budget it was given.

    A partial answer from an interrupted run is not the answer to the problem,
    so the solvers raise instead of returning one. Callers that set a budget are
    expected to catch this and record that the run did not finish.
    """


class Deadline:
    """A wall clock budget that a solver can check as it works.

    check() is cheap but not free, so hot loops call it once every few thousand
    operations through tick(). Every counter starts fresh, so one Deadline can
    be shared by several solvers if the caller wants a total budget.
    """

    __slots__ = ("limit", "started", "counter", "interval")

    def __init__(self, seconds: Optional[float] = None, interval: int = 2048):
        self.limit = seconds
        self.started = time.perf_counter()
        self.counter = 0
        self.interval = interval

    @property
    def enabled(self) -> bool:
        return self.limit is not None and self.limit > 0

    def expired(self) -> bool:
        return self.enabled and (time.perf_counter() - self.started) > self.limit

    def tick(self) -> None:
        """Count one unit of work and raise once the budget is gone."""
        if not self.enabled:
            return
        self.counter += 1
        if self.counter >= self.interval:
            self.counter = 0
            if self.expired():
                raise TimeoutError_(f"time budget of {self.limit:g} seconds ran out")


@dataclass
class FlowResult:
    """What every max flow solver returns.

    value: total flow leaving the source.
    flow: sparse map (u, v) -> flow on the original edge u -> v.
    augmentations: number of augmenting paths or pushes that changed the flow.
    iterations: algorithm specific work counter (BFS rounds, relabels, and so on).
    stats: extra measurements such as bottlenecks, path lengths, or profiles.
    """

    value: float
    flow: Dict[Tuple[int, int], float] = field(default_factory=dict)
    algorithm: str = ""
    augmentations: int = 0
    iterations: int = 0
    stats: Dict[str, object] = field(default_factory=dict)

    def to_matrix(self, n: Optional[int] = None) -> np.ndarray:
        """Dense flow matrix. Use only on small graphs."""
        if n is None:
            n = 0 if not self.flow else max(max(u, v) for u, v in self.flow) + 1
        mat = np.zeros((n, n), dtype=float)
        for (u, v), f in self.flow.items():
            mat[u, v] += f
        return mat

    def edge_flow(self, u: int, v: int) -> float:
        return self.flow.get((u, v), 0.0)


@dataclass
class Edge:
    """Read only view of one edge, used by printers and tests."""

    index: int
    u: int
    v: int
    capacity: float
    cost: float
    flow: float


class FlowNetwork:
    """Directed graph with residual capacities, costs, and reverse edges."""

    __slots__ = ("n", "_to", "_cap", "_orig_cap", "_cost", "_adj", "_names", "source", "sink")

    def __init__(self, n: int, source: Optional[int] = None, sink: Optional[int] = None):
        if n < 0:
            raise ValueError("node count must be non negative")
        self.n = int(n)
        self._to: List[int] = []
        self._cap: List[float] = []
        self._orig_cap: List[float] = []
        self._cost: List[float] = []
        self._adj: List[List[int]] = [[] for _ in range(self.n)]
        self._names: Dict[int, str] = {}
        self.source = source
        self.sink = sink

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add_node(self, name: Optional[str] = None) -> int:
        """Grow the network by one node and return its id."""
        node = self.n
        self.n += 1
        self._adj.append([])
        if name is not None:
            self._names[node] = name
        return node

    def set_name(self, node: int, name: str) -> None:
        self._names[node] = name

    def name_of(self, node: int) -> str:
        return self._names.get(node, str(node))

    def add_edge(self, u: int, v: int, capacity: float, cost: float = 0.0) -> int:
        """Add edge u -> v and its reverse edge. Returns the forward edge index.

        The reverse edge starts with capacity 0 and cost -cost. Pushing f units
        along the forward edge raises the reverse capacity by f, which is what
        makes cancelling a bad decision possible.
        """
        if u == v:
            raise ValueError("self loops are not supported")
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError(f"edge endpoint out of range: {u} -> {v}")
        if capacity < 0:
            raise ValueError("capacity must be non negative")
        index = len(self._to)
        self._to.append(v)
        self._cap.append(capacity)
        self._orig_cap.append(capacity)
        self._cost.append(cost)
        self._adj[u].append(index)

        self._to.append(u)
        self._cap.append(0.0)
        self._orig_cap.append(0.0)
        self._cost.append(-cost)
        self._adj[v].append(index + 1)
        return index

    def add_edges_from(self, edges: Iterable[Sequence]) -> None:
        for edge in edges:
            if len(edge) == 3:
                u, v, cap = edge
                self.add_edge(u, v, cap)
            elif len(edge) == 4:
                u, v, cap, cost = edge
                self.add_edge(u, v, cap, cost)
            else:
                raise ValueError(f"bad edge tuple: {edge!r}")

    def reset(self) -> None:
        """Restore every residual capacity to its original value."""
        for i, cap in enumerate(self._orig_cap):
            self._cap[i] = cap

    def copy(self) -> "FlowNetwork":
        other = FlowNetwork(self.n, self.source, self.sink)
        other._to = list(self._to)
        other._cap = list(self._cap)
        other._orig_cap = list(self._orig_cap)
        other._cost = list(self._cost)
        other._adj = [list(a) for a in self._adj]
        other._names = dict(self._names)
        return other

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n

    @property
    def edge_count(self) -> int:
        """Number of directed edges including reverse edges."""
        return len(self._to)

    @property
    def original_edge_count(self) -> int:
        return len(self._to) // 2

    @property
    def capacities(self) -> List[float]:
        return self._cap

    @property
    def heads(self) -> List[int]:
        return self._to

    @property
    def costs(self) -> List[float]:
        return self._cost

    @property
    def adjacency(self) -> List[List[int]]:
        return self._adj

    def iter_edges(self) -> Iterator[Edge]:
        for i in range(0, len(self._to), 2):
            u = self._to[i ^ 1]
            yield Edge(
                index=i,
                u=u,
                v=self._to[i],
                capacity=self._orig_cap[i],
                cost=self._cost[i],
                flow=self._orig_cap[i] - self._cap[i],
            )

    def residual_capacity(self, u: int, v: int) -> float:
        total = 0.0
        for i in self._adj[u]:
            if self._to[i] == v:
                total += self._cap[i]
        return total

    def get_residual_graph(self) -> Tuple[int, List[Tuple[int, int, float]]]:
        """Return (n, edges) for the current residual graph.

        Only edges with positive residual capacity are listed.
        """
        edges: List[Tuple[int, int, float]] = []
        for u in range(self.n):
            for i in self._adj[u]:
                if self._cap[i] > 0:
                    edges.append((u, self._to[i], self._cap[i]))
        return self.n, edges

    def residual_to_networkx(self, min_cap: float = 0.0):
        """Residual graph as a networkx DiGraph, for plotting only."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        for u in range(self.n):
            for i in self._adj[u]:
                if self._cap[i] > min_cap:
                    graph.add_edge(u, self._to[i], capacity=self._cap[i])
        return graph

    def to_networkx(self):
        """Original edges as a networkx DiGraph keyed by edge index."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        for edge in self.iter_edges():
            graph.add_edge(edge.u, edge.v, capacity=edge.capacity, cost=edge.cost, index=edge.index)
        return graph

    def out_capacity(self, u: int) -> float:
        return sum(self._orig_cap[i] for i in self._adj[u] if i % 2 == 0)

    def in_capacity(self, v: int) -> float:
        return sum(self._orig_cap[i ^ 1] for i in self._adj[v] if i % 2 == 1)

    # ------------------------------------------------------------------
    # Flow helpers
    # ------------------------------------------------------------------
    def flow_map(self) -> Dict[Tuple[int, int], float]:
        """Sparse flow map built from the current residual capacities."""
        flow: Dict[Tuple[int, int], float] = {}
        # Parallel edges are legal (the segmentation seeds add one), so flows on
        # the same ordered pair are summed, not overwritten.
        for i in range(0, len(self._to), 2):
            used = self._orig_cap[i] - self._cap[i]
            if used > 1e-12:
                key = (self._to[i ^ 1], self._to[i])
                flow[key] = flow.get(key, 0.0) + used
        return flow

    def flow_balance(self, flow: Dict[Tuple[int, int], float]) -> Dict[int, float]:
        """Net flow out of each node: out flow minus in flow."""
        balance = {u: 0.0 for u in range(self.n)}
        for (u, v), f in flow.items():
            balance[u] += f
            balance[v] -= f
        return balance

    def conservation_error(self, flow: Dict[Tuple[int, int], float], s: int, t: int) -> float:
        """Largest violation of conservation at any node except s and t."""
        balance = self.flow_balance(flow)
        worst = 0.0
        for node, value in balance.items():
            if node in (s, t):
                continue
            worst = max(worst, abs(value))
        return worst

    def capacity_violations(self, flow: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float, float]]:
        """Edges where the flow is negative or above capacity."""
        limits: Dict[Tuple[int, int], float] = {}
        for edge in self.iter_edges():
            key = (edge.u, edge.v)
            limits[key] = limits.get(key, 0.0) + edge.capacity
        bad = []
        for (u, v), f in flow.items():
            cap = limits.get((u, v), 0.0)
            if f < -1e-9 or f > cap + 1e-9:
                bad.append((u, v, f, cap))
        return bad

    def reachable_from(self, s: int, flow: Optional[Dict[Tuple[int, int], float]] = None) -> set:
        """Nodes reachable from s in the residual graph.

        If flow is given, the reachable set is computed from that flow using the
        original capacities, so a separate residual copy is not needed.
        """
        seen = {s}
        stack = [s]
        if flow is None:
            while stack:
                u = stack.pop()
                for i in self._adj[u]:
                    if self._cap[i] > 0:
                        v = self._to[i]
                        if v not in seen:
                            seen.add(v)
                            stack.append(v)
            return seen

        caps: Dict[Tuple[int, int], float] = {}
        for edge in self.iter_edges():
            key = (edge.u, edge.v)
            caps[key] = caps.get(key, 0.0) + edge.capacity
        while stack:
            u = stack.pop()
            for i in self._adj[u]:
                if i % 2 == 0:
                    v = self._to[i]
                    if caps.get((u, v), 0.0) - flow.get((u, v), 0.0) > 1e-12 and v not in seen:
                        seen.add(v)
                        stack.append(v)
                else:
                    v = self._to[i]
                    if flow.get((v, u), 0.0) > 1e-12 and v not in seen:
                        seen.add(v)
                        stack.append(v)
        return seen

    def min_cut(self, s: int, flow: Dict[Tuple[int, int], float]) -> Tuple[set, set, float]:
        """Return (source side, sink side, cut capacity) for a given flow."""
        src_side = self.reachable_from(s, flow)
        sink_side = set(range(self.n)) - src_side
        cut = 0.0
        for edge in self.iter_edges():
            if edge.u in src_side and edge.v in sink_side:
                cut += edge.capacity
        return src_side, sink_side, cut

    def __repr__(self) -> str:
        return (
            f"FlowNetwork(n={self.n}, edges={self.original_edge_count}, "
            f"source={self.source}, sink={self.sink})"
        )
