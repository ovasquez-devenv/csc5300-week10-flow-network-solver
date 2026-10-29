"""Seeded generators for every graph family the project benchmarks.

Every function takes a seed and builds its own numpy Generator or random.Random
from it. Two calls with the same seed produce the same graph, which is what
makes the benchmark numbers repeatable.

Graph families:

- random_sparse: layered graph, each node points forward to a few others. This
  is the shape of a realistic network flow instance and it keeps paths short.
- random_dense: layered graph with most forward pairs present.
- layered: the shared builder, with a fan out parameter.
- bipartite_matching: left and right sides with random edges.
- grid_graph: a 2D grid, the shape used for image segmentation.
- capacity_distribution: uniform, exponential, or power law edge capacities.

The layered shape matters for the benchmark. A purely random digraph with a
source and sink tends to give one tiny cut and a trivial answer, while layers
force flow to route through the middle and produce work for the solvers.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.network import FlowNetwork


def _capacities(rng: np.random.Generator, count: int, distribution: str, low: float, high: float) -> np.ndarray:
    """Draw count capacities from the requested distribution."""
    if distribution == "uniform":
        return rng.integers(int(low), int(high) + 1, size=count).astype(float)
    if distribution == "exponential":
        scale = max(1.0, (high - low) / 4.0)
        values = rng.exponential(scale, size=count) + low
        return np.maximum(1.0, np.round(values))
    if distribution == "power_law":
        # Pareto tail: a few very large edges, most edges small.
        values = low + (rng.pareto(1.6, size=count) + 1.0) * max(1.0, (high - low) / 8.0)
        return np.clip(np.round(values), 1, high * 4)
    raise ValueError(f"unknown capacity distribution: {distribution}")


def layered_network(
    layers: int,
    width: int,
    fan_out: int,
    seed: int = 5300,
    distribution: str = "uniform",
    low: float = 1.0,
    high: float = 100.0,
    cost_range: Tuple[float, float] = (1.0, 10.0),
) -> FlowNetwork:
    """Build a layered graph with a single source and a single sink.

    Layer 0 is the source, the last layer is the sink, and the layers between
    hold width nodes each. Every node points forward to up to fan_out nodes in
    the next layer, chosen without replacement from a seeded generator.
    """
    if layers < 2:
        raise ValueError("need at least two layers")
    if width < 1:
        raise ValueError("width must be at least 1")

    rng = np.random.default_rng(seed)
    middle = layers - 2
    node_ids: List[List[int]] = [[] for _ in range(layers)]

    net = FlowNetwork(0)
    node_ids[0] = [net.add_node(f"source")]
    for layer in range(1, layers - 1):
        node_ids[layer] = [net.add_node(f"L{layer}_{i}") for i in range(width)]
    node_ids[layers - 1] = [net.add_node("sink")]

    net.source = node_ids[0][0]
    net.sink = node_ids[layers - 1][0]

    edge_list: List[Tuple[int, int, float, float]] = []
    c_low, c_high = cost_range
    for layer in range(layers - 1):
        left = node_ids[layer]
        right = node_ids[layer + 1]
        # The source and the sink are joined to the full first and last layer.
        # With a fan out of two, the source would cap the whole answer at a
        # couple of edges and the cut would never see the middle of the graph.
        # A wide source is also how a real network looks: many entry points
        # feeding a few trunk lines.
        wide = layer == 0 or layer == layers - 2
        for u in left:
            count = len(right) if wide else min(len(right), fan_out)
            if count >= len(right):
                targets = range(len(right))
            else:
                targets = rng.choice(len(right), size=count, replace=False)
            for t in targets:
                v = right[int(t)]
                cap = float(_capacities(rng, 1, distribution, low, high)[0])
                cost = float(rng.integers(int(c_low), int(c_high) + 1))
                edge_list.append((u, v, cap, cost))
    net.add_edges_from(edge_list)
    return net


def random_sparse(
    nodes: int = 1000,
    seed: int = 5300,
    fan_out: int = 2,
    distribution: str = "uniform",
    low: float = 1.0,
    high: float = 100.0,
) -> FlowNetwork:
    """Sparse layered graph with roughly nodes nodes and a small edge count."""
    layers = max(2, int(round(nodes ** 0.5)))
    width = max(1, (nodes - 2) // max(1, layers - 2))
    return layered_network(layers, width, fan_out, seed, distribution, low, high)


def random_dense(
    nodes: int = 1000,
    seed: int = 5300,
    distribution: str = "uniform",
    low: float = 1.0,
    high: float = 100.0,
    fan_out: int = 8,
) -> FlowNetwork:
    """Dense layered graph: each node reaches up to fan_out nodes ahead.

    Sparse uses a fan out of 2, so a fan out of 8 is four times as many edges at
    the same node count. A fully connected layered graph would be quadratic in
    the width and would put 11 million edges on the 50,000 node case, which no
    max flow solver finishes in a reasonable time and which is not a realistic
    workload. Eight forward edges per node is dense enough to change the shape
    of the search while staying inside the runtime budget.
    """
    layers = max(2, int(round(nodes ** 0.5)))
    width = max(1, (nodes - 2) // max(1, layers - 2))
    effective = max(2, min(width, fan_out))
    return layered_network(layers, width, effective, seed, distribution, low, high)


def random_digraph(
    nodes: int = 50,
    edge_probability: float = 0.15,
    seed: int = 5300,
    low: float = 1.0,
    high: float = 20.0,
) -> FlowNetwork:
    """Plain random digraph with node 0 as source and node n - 1 as sink.

    Only forward edges (u < v) are added, so the graph is acyclic and there is
    always a well defined answer. This is the family the tests use, because a
    small acyclic graph is easy to check exhaustively.
    """
    rng = np.random.default_rng(seed)
    net = FlowNetwork(nodes, 0, nodes - 1)
    for u in range(nodes):
        for v in range(u + 1, nodes):
            if rng.random() < edge_probability:
                net.add_edge(u, v, float(rng.integers(int(low), int(high) + 1)))
    return net


def bipartite_matching_graph(
    left: int = 20,
    right: int = 20,
    degree: int = 3,
    seed: int = 5300,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """A bipartite instance with a bounded number of edges per left node."""
    rng = random.Random(seed)
    edges: List[Tuple[int, int]] = []
    for u in range(left):
        count = min(right, degree)
        for v in rng.sample(range(right), count):
            edges.append((u, v))
    return left, right, edges


def grid_graph(
    rows: int,
    cols: int,
    seed: int = 5300,
    similarity_scale: float = 40.0,
    source_weight: float = 1.0,
    sink_weight: float = 1.0,
) -> FlowNetwork:
    """2D grid with 4 neighbor edges, plus a source and a sink.

    Neighbor capacities fall off with a seeded random difference, which stands
    in for how similar two adjacent pixels are. The source and sink edges are
    the unary terms. This is the image segmentation shape without the image.
    """
    rng = np.random.default_rng(seed)
    pixels = rows * cols
    source = pixels
    sink = pixels + 1
    net = FlowNetwork(pixels + 2, source, sink)
    values = rng.random(pixels)

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                w = float(np.exp(-similarity_scale * (values[u] - values[u + 1]) ** 2))
                net.add_edge(u, u + 1, max(1e-3, w))
                net.add_edge(u + 1, u, max(1e-3, w))
            if r + 1 < rows:
                w = float(np.exp(-similarity_scale * (values[u] - values[u + cols]) ** 2))
                net.add_edge(u, u + cols, max(1e-3, w))
                net.add_edge(u + cols, u, max(1e-3, w))
            net.add_edge(source, u, float(source_weight * values[u]))
            net.add_edge(u, sink, float(sink_weight * (1.0 - values[u])))
    return net


def random_costs(net: FlowNetwork, seed: int = 5300, low: int = 1, high: int = 20) -> None:
    """Assign random costs to every forward edge of an existing network."""
    rng = np.random.default_rng(seed)
    costs = net.costs
    for i in range(0, len(costs), 2):
        costs[i] = float(rng.integers(low, high + 1))
        costs[i + 1] = -costs[i]


def capacity_summary(net: FlowNetwork) -> Dict[str, float]:
    """Basic counts and capacity statistics, useful in benchmark output."""
    caps = [e.capacity for e in net.iter_edges()]
    if not caps:
        return {"edges": 0, "total_capacity": 0.0, "mean_capacity": 0.0, "max_capacity": 0.0}
    return {
        "nodes": net.n,
        "edges": len(caps),
        "total_capacity": float(sum(caps)),
        "mean_capacity": float(sum(caps)) / len(caps),
        "max_capacity": float(max(caps)),
    }
