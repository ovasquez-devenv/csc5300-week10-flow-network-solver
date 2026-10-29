"""Image segmentation by graph cuts.

This is the Boykov and Jolly formulation. Every pixel becomes a node. Edges
between neighboring pixels carry a capacity equal to how similar the two pixels
are, so cutting a strong similarity edge is expensive and the cut prefers to
separate pixels that look different. Each pixel also gets an edge to a source
node (foreground) and to a sink node (background), with capacity equal to how
strongly that pixel looks like foreground or background.

A minimum s-t cut then splits the graph into a source side, which becomes the
foreground mask, and a sink side, which becomes the background. Writing the cut
as a sum over edges shows the three terms: the penalty for a foreground pixel
that landed on the background side, the penalty for the reverse, and the penalty
for separating two similar neighbors. Minimizing that sum is exactly what the
cut does.

The likelihoods here come from a Gaussian fit to seed pixels, which keeps the
project dependency free and makes the result easy to explain. Nothing about the
graph cut part depends on that choice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import FlowNetwork


def _gaussian_log_likelihood(values: np.ndarray, mean: float, var: float) -> float:
    """Log likelihood of a pixel value under a 1D Gaussian, clipped to stay finite."""
    var = max(var, 1e-6)
    return -0.5 * ((values - mean) ** 2) / var - 0.5 * np.log(2 * np.pi * var)


def make_test_image(
    size: int = 64,
    seed: int = 5300,
    brightness: float = 0.6,
    noise: float = 0.06,
) -> np.ndarray:
    """A grayscale image with one bright rectangle on a darker background.

    The rectangle is the thing segmentation should find. Noise is Gaussian and
    seeded, so the same seed always gives the same image.
    """
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 0.3, dtype=float)
    lo = size // 4
    hi = size - lo
    image[lo:hi, lo:hi] = brightness
    image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0)


def segment_image(
    image: np.ndarray,
    foreground_seeds: Optional[Sequence[Tuple[int, int]]] = None,
    background_seeds: Optional[Sequence[Tuple[int, int]]] = None,
    similarity_scale: float = 40.0,
    hard_seed_capacity: float = 1e6,
    solver: str = "pr",
) -> Dict[str, object]:
    """Segment a grayscale image into foreground and background.

    image is a 2D array of floats. foreground_seeds and background_seeds are
    (row, col) pairs that are known to be one or the other. When they are
    omitted, rough quadrants are used: the border is background and the center
    block is foreground, which is enough for the rectangle test image.

    similarity_scale controls how sharply the neighbor capacity falls off with a
    difference in brightness. Bigger values mean neighbor edges matter less and
    the seeds dominate; smaller values mean the cut hugs image edges harder.

    Returns a dict with the mask, the flow value, the cut capacity, and the
    network size.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    rows, cols = image.shape
    pixels = rows * cols

    if foreground_seeds is None:
        r0, r1 = rows // 4, rows - rows // 4
        c0, c1 = cols // 4, cols - cols // 4
        foreground_seeds = [(r, c) for r in range(r0, r1, 4) for c in range(c0, c1, 4)]
    if background_seeds is None:
        background_seeds = (
            [(0, c) for c in range(cols)]
            + [(rows - 1, c) for c in range(cols)]
            + [(r, 0) for r in range(rows)]
            + [(r, cols - 1) for r in range(rows)]
        )

    flat = image.reshape(-1)
    fg_values = np.array([image[r, c] for r, c in foreground_seeds], dtype=float)
    bg_values = np.array([image[r, c] for r, c in background_seeds], dtype=float)
    fg_mean, fg_var = float(fg_values.mean()), float(fg_values.var())
    bg_mean, bg_var = float(bg_values.mean()), float(bg_values.var())

    source = pixels
    sink = pixels + 1
    net = FlowNetwork(pixels + 2, source, sink)

    # Neighbor edges: capacity falls off as the two pixels differ more.
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                w = float(np.exp(-similarity_scale * abs(flat[u] - flat[u + 1]) ** 2))
                net.add_edge(u, u + 1, w)
                net.add_edge(u + 1, u, w)
            if r + 1 < rows:
                w = float(np.exp(-similarity_scale * abs(flat[u] - flat[u + cols]) ** 2))
                net.add_edge(u, u + cols, w)
                net.add_edge(u + cols, u, w)

    # Source and sink edges from the Gaussian likelihoods. Only one edge per
    # pixel is needed. If a pixel looks more like foreground, a source edge of
    # that much weight is cut when the pixel ends up on the background side. If
    # it looks more like background, a sink edge is cut when it ends up in the
    # foreground. That pair of cases gives exactly the unary cost of the cut and
    # avoids the constant offset that adding both edges would introduce.
    fg_ll = _gaussian_log_likelihood(flat, fg_mean, fg_var)
    bg_ll = _gaussian_log_likelihood(flat, bg_mean, bg_var)
    diff = fg_ll - bg_ll
    for p in range(pixels):
        d = float(diff[p])
        if d > 0:
            net.add_edge(source, p, d)
        elif d < 0:
            net.add_edge(p, sink, -d)

    # Hard seeds: a huge edge pins the pixel to its side.
    for r, c in foreground_seeds:
        net.add_edge(source, r * cols + c, hard_seed_capacity)
    for r, c in background_seeds:
        net.add_edge(r * cols + c, sink, hard_seed_capacity)

    if solver == "ek":
        result = edmonds_karp(net, source, sink)
        cut = result.value
    elif solver == "mcf":
        result = min_cost_flow(net, source, sink)
        cut = result.value
    elif solver == "pr":
        from core.push_relabel import push_relabel

        result = push_relabel(net, source, sink)
        cut = result.value
    else:
        raise ValueError(f"unknown solver: {solver}")

    src_side, sink_side, cut_capacity = net.min_cut(source, result.flow)
    mask = np.zeros(pixels, dtype=bool)
    for p in range(pixels):
        mask[p] = p in src_side
    mask = mask.reshape(rows, cols)

    true_mask = image > (fg_mean + bg_mean) / 2.0
    agreement = float((mask == true_mask).mean())

    return {
        "mask": mask,
        "flow": result.value,
        "cut_capacity": cut_capacity,
        "foreground_mean": fg_mean,
        "background_mean": bg_mean,
        "shape": (rows, cols),
        "nodes": net.n,
        "edges": net.original_edge_count,
        "agreement_with_threshold": agreement,
        "solver": solver,
    }


def mask_accuracy(mask: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of pixels that match the ground truth mask."""
    if mask.shape != truth.shape:
        raise ValueError("mask and truth must have the same shape")
    return float((mask.astype(bool) == truth.astype(bool)).mean())
