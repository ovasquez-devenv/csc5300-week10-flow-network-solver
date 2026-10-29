"""Static pictures of a flow network.

Two jobs here. Draw the graph with capacities and flows labeled, and draw the
cut that the flow induces. Both write straight to a file with the Agg backend,
so they work with no display at all, which is how the benchmark runs.

Node positions come from one of three layouts: layered (by distance from the
source), spring (networkx force directed), or circular. Layered is the default
because the graphs this project builds are layered, and a layered drawing makes
the source to sink structure obvious.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

from core.network import FlowNetwork

DEFAULT_DPI = 150


def _finish(fig, path: str, dpi: int = DEFAULT_DPI) -> str:
    """Save with a tight box and a folder that exists, then close the figure."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def layered_layout(net: FlowNetwork, source: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """Place nodes in columns by how far they sit along the graph.

    The first pass is a breadth first search from the source. Nodes the source
    cannot reach would otherwise land in one meaningless pile at the far right,
    so a second pass pushes them one column past their furthest predecessor,
    repeating until nothing moves. On a layered graph that puts every node in
    the column it belongs to, reachable or not.
    """
    s = net.source if source is None else source
    layer: Dict[int, int] = {s: 0}
    frontier = [s]
    while frontier:
        nxt = []
        for u in frontier:
            for i in net.adjacency[u]:
                if i % 2 == 0:
                    v = net.heads[i]
                    if v not in layer:
                        layer[v] = layer[u] + 1
                        nxt.append(v)
        frontier = nxt

    # Longest path placement for the rest, bounded so a cycle cannot loop.
    unplaced = [v for v in range(net.n) if v not in layer]
    for _ in range(net.n):
        moved = False
        for v in unplaced:
            if v in layer:
                continue
            best = None
            for i in net.adjacency[v]:
                if i % 2 == 1:
                    u = net.heads[i]
                    if u in layer:
                        cand = layer[u] + 1
                        best = cand if best is None else max(best, cand)
            if best is not None:
                layer[v] = best
                moved = True
        if not moved:
            break
    for v in range(net.n):
        layer.setdefault(v, 0)

    columns: Dict[int, List[int]] = {}
    for v in range(net.n):
        columns.setdefault(layer[v], []).append(v)

    pos: Dict[int, Tuple[float, float]] = {}
    for depth in sorted(columns):
        nodes = sorted(columns[depth])
        count = len(nodes)
        for k, v in enumerate(nodes):
            y = 0.0 if count == 1 else (k - (count - 1) / 2.0) / max(1, count - 1)
            pos[v] = (float(depth), y)
    return pos


def network_layout(net: FlowNetwork, kind: str = "layered", seed: int = 5300) -> Dict[int, Tuple[float, float]]:
    """Pick a layout by name. Falls back to layered for unknown names."""
    if kind == "layered":
        return layered_layout(net)
    graph = net.to_networkx()
    if kind == "spring":
        return nx.spring_layout(graph, seed=seed)
    if kind == "circular":
        return nx.circular_layout(graph)
    return layered_layout(net)


def draw_network(
    net: FlowNetwork,
    path: str,
    flow: Optional[Dict[Tuple[int, int], float]] = None,
    layout: str = "layered",
    title: str = "Flow network",
    show_capacities: bool = True,
    show_flows: bool = True,
    max_labeled_edges: int = 40,
    node_size: float = 900.0,
    figsize: Tuple[float, float] = (12.0, 7.0),
    dpi: int = DEFAULT_DPI,
) -> str:
    """Draw the network with capacity and flow labels and save it.

    Only the first max_labeled_edges edges get text labels, because a graph with
    thousands of edges turns into unreadable ink. The picture still shows the
    whole graph; the labels are what gets limited.
    """
    graph = nx.DiGraph()
    for v in range(net.n):
        graph.add_node(v, label=net.name_of(v))
    for edge in net.iter_edges():
        graph.add_edge(edge.u, edge.v, capacity=edge.capacity, cost=edge.cost)

    pos = network_layout(net, layout)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color="#cfe3f7", edgecolors="#20486f")
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=8,
        labels={v: net.name_of(v) for v in graph.nodes},
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12,
        edge_color="#4a4a4a", width=1.0, node_size=node_size, connectionstyle="arc3,rad=0.06",
    )

    if show_capacities or show_flows:
        edges = list(net.iter_edges())[:max_labeled_edges]
        for edge in edges:
            x0, y0 = pos[edge.u]
            x1, y1 = pos[edge.v]
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            parts = []
            if show_capacities:
                parts.append(f"c={edge.capacity:g}")
            if show_flows and flow is not None:
                parts.append(f"f={flow.get((edge.u, edge.v), 0.0):g}")
            if parts:
                ax.text(xm, ym, "\n".join(parts), fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_title(title)
    ax.set_xlabel("layer (hops from source)")
    ax.set_ylabel("position within layer")
    ax.axis("off")
    fig.tight_layout()
    return _finish(fig, path, dpi)


def draw_cut(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    path: str,
    source: Optional[int] = None,
    layout: str = "layered",
    title: str = "Min cut from the flow",
    figsize: Tuple[float, float] = (12.0, 7.0),
    dpi: int = DEFAULT_DPI,
) -> str:
    """Shade the two sides of the min cut and highlight the cut edges.

    The source side is the set of nodes still reachable from the source in the
    residual graph. The edges that leave it are the cut, and their total
    capacity equals the flow value.
    """
    s = net.source if source is None else source
    left, right, cut_value = net.min_cut(s, flow)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    for edge in net.iter_edges():
        graph.add_edge(edge.u, edge.v)
    pos = network_layout(net, layout)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=10,
        edge_color="#bbbbbb", width=0.8, connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=sorted(left), node_size=600,
                           node_color="#8fd18f", edgecolors="#2b6b2b", label="source side")
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=sorted(right), node_size=600,
                           node_color="#f2a6a6", edgecolors="#7d2a2a", label="sink side")

    cut_edges = [(e.u, e.v) for e in net.iter_edges() if e.u in left and e.v in right]
    if cut_edges:
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edgelist=cut_edges, arrows=True, arrowstyle="-|>",
            arrowsize=14, edge_color="#c1121f", width=2.4,
        )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)

    ax.set_title(f"{title} (cut capacity = {cut_value:g})")
    ax.set_xlabel("layer (hops from source)")
    ax.set_ylabel("position within layer")
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    return _finish(fig, path, dpi)


def draw_matching(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
    pairs: Sequence[Tuple[int, int]],
    path: str,
    title: str = "Bipartite matching",
    figsize: Tuple[float, float] = (10.0, 7.0),
    dpi: int = DEFAULT_DPI,
) -> str:
    """Two columns of nodes with the allowed edges light and the pairs dark."""
    fig, ax = plt.subplots(figsize=figsize)
    left_y = {u: -(u - (n_left - 1) / 2.0) for u in range(n_left)}
    right_y = {v: -(v - (n_right - 1) / 2.0) for v in range(n_right)}
    matched = set(pairs)

    for u, v in edges:
        chosen = (u, v) in matched
        ax.plot(
            [0.0, 1.0], [left_y[u], right_y[v]],
            color="#c1121f" if chosen else "#cccccc",
            linewidth=2.6 if chosen else 0.8,
            zorder=3 if chosen else 1,
        )
    for u in range(n_left):
        ax.scatter([0.0], [left_y[u]], s=420, c="#cfe3f7", edgecolors="#20486f", zorder=4)
        ax.text(0.0, left_y[u], str(u), ha="center", va="center", fontsize=8, zorder=5)
    for v in range(n_right):
        ax.scatter([1.0], [right_y[v]], s=420, c="#f7e3cf", edgecolors="#7d4f20", zorder=4)
        ax.text(1.0, right_y[v], str(v), ha="center", va="center", fontsize=8, zorder=5)

    ax.set_xlim(-0.25, 1.25)
    ax.set_title(f"{title} ({len(pairs)} pairs)")
    ax.set_xlabel("side")
    ax.set_ylabel("node index")
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["left", "right"])
    ax.axis("off")
    fig.tight_layout()
    return _finish(fig, path, dpi)


def draw_segmentation(
    image,
    mask,
    path: str,
    title: str = "Graph cut segmentation",
    figsize: Tuple[float, float] = (10.0, 3.6),
    dpi: int = DEFAULT_DPI,
) -> str:
    """Original image, the mask, and the mask laid over the image."""
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("input image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("foreground mask")
    overlay = np.zeros((*mask.shape, 3))
    overlay[..., 0] = np.where(mask, 1.0, image)
    overlay[..., 1] = np.where(mask, 0.35, image)
    overlay[..., 2] = np.where(mask, 0.35, image)
    axes[2].imshow(overlay)
    axes[2].set_title("mask over image")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path, dpi)
