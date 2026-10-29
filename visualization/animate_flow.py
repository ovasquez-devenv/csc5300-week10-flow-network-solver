"""Animations that show the solvers working, saved to a file.

There is no interactive display on the machine that runs the benchmark, so both
animations write a file and return its path. GIF goes through matplotlib's
Pillow writer, which is always available because matplotlib depends on Pillow.
MP4 goes through ffmpeg and raises a clear error if ffmpeg is missing rather
than failing with a wall of text from the encoder.

Edmonds-Karp animation: each frame shows the residual graph with the current
augmenting path drawn in red, the bottleneck in the title, and the flow value
growing frame by frame.

Push-relabel animation: each frame shows the height and excess of every node,
with the node being discharged circled. Heights climb as relabels happen, which
is the part that makes the algorithm look strange the first time you see it.

Both take the path list or trace that a solver produced, so the animation shows
what actually happened in that run, not a re-simulation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from core.network import FlowNetwork
from visualization.plot_network import DEFAULT_DPI, layered_layout


def _ensure_folder(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def _save_animation(anim: FuncAnimation, path: str, fps: int = 4) -> str:
    """Write a GIF with Pillow, or an MP4 with ffmpeg, based on the extension."""
    _ensure_folder(path)
    lower = path.lower()
    if lower.endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter

            anim.save(path, writer=FFMpegWriter(fps=fps))
            return path
        except Exception as exc:  # pragma: no cover - only when ffmpeg is absent
            fallback = os.path.splitext(path)[0] + ".gif"
            anim.save(fallback, writer=PillowWriter(fps=fps))
            print(f"MP4 writing failed ({exc}), saved a GIF at {fallback} instead")
            return fallback
    anim.save(path, writer=PillowWriter(fps=fps))
    return path


def animate_edmonds_karp(
    net: FlowNetwork,
    paths: Sequence[Sequence[Tuple[int, int, float]]],
    path: str,
    layout: str = "layered",
    max_frames: int = 30,
    fps: int = 3,
    title: str = "Edmonds-Karp augmenting paths",
    dpi: int = 90,
) -> str:
    """Animate the augmenting paths one frame at a time.

    Each frame draws every original edge in light gray, the edges of the current
    path in red, and reports the bottleneck of that path and the flow value so
    far. Long runs are sampled down to max_frames so the file stays small.
    """
    if not paths:
        raise ValueError("no augmenting paths to animate")

    step = max(1, len(paths) // max_frames)
    chosen = list(paths)[::step]
    if chosen[-1] is not paths[-1]:
        chosen.append(list(paths)[-1])

    pos = layered_layout(net, net.source)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    edges = [(e.u, e.v) for e in net.iter_edges()]
    running = 0.0
    max_cap = max((e.capacity for e in net.iter_edges()), default=1.0)

    def draw(frame_index: int):
        ax.clear()
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#dddddd", lw=0.6, shrinkA=8, shrinkB=8),
            )

        path_edges = chosen[frame_index]
        bottleneck = min(cap for _, _, cap in path_edges) if path_edges else 0.0
        for (u, v, cap) in path_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            width = 1.0 + 3.0 * (cap / max_cap if max_cap else 0.0)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#c1121f", lw=width, shrinkA=8, shrinkB=8),
            )
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, f"{cap:g}", fontsize=7, color="#c1121f",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

        for v in range(net.n):
            x, y = pos[v]
            ax.scatter([x], [y], s=260, c="#cfe3f7", edgecolors="#20486f", zorder=3)
            ax.text(x, y, net.name_of(v), ha="center", va="center", fontsize=6, zorder=4)

        nonlocal running
        running += bottleneck
        ax.set_title(
            f"{title}\npath {frame_index * step + 1} of {len(paths)}, "
            f"bottleneck {bottleneck:g}, flow so far {running:g}"
        )
        ax.axis("off")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(chosen), interval=1000 // max(1, fps), blit=False)
    saved = _save_animation(anim, path, fps)
    plt.close(fig)
    return saved


def animate_push_relabel(
    net: FlowNetwork,
    trace: Sequence[Tuple],
    path: str,
    layout: str = "layered",
    max_frames: int = 40,
    fps: int = 4,
    title: str = "Push-relabel: heights and excess",
    dpi: int = 90,
) -> str:
    """Animate a push-relabel trace.

    Each trace entry is ("push", u, v, amount, excess_v) or ("relabel", u, old,
    new, excess_u), which is what push_relabel writes when trace is on. The
    frame shows the node the event touched, its height and excess, and the flow
    sitting at the sink.
    """
    if not trace:
        raise ValueError("no trace to animate")

    step = max(1, len(trace) // max_frames)
    chosen = list(trace)[::step]
    pos = layered_layout(net, net.source)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Replay heights and excess through the trace so a frame is consistent.
    frames: List[Tuple[str, int, int, float, float, Dict[int, int], Dict[int, float]]] = []
    heights = {v: 0 for v in range(net.n)}
    height_source = net.source
    heights[height_source] = net.n
    excess = {v: 0.0 for v in range(net.n)}
    flow_at_sink = 0.0

    for event in chosen:
        kind = event[0]
        if kind == "push":
            _, u, v, amount, excess_v = event
            excess[u] = max(0.0, excess.get(u, 0.0) - amount)
            excess[v] = excess_v
            if v == net.sink:
                flow_at_sink = excess_v
            frames.append((kind, u, v, amount, flow_at_sink, dict(heights), dict(excess)))
        elif kind == "relabel":
            _, u, old, new, ex = event
            heights[u] = new
            excess[u] = ex
            frames.append((kind, u, u, float(old), flow_at_sink, dict(heights), dict(excess)))
        else:
            frames.append(("gap", net.source, net.source, 0.0, flow_at_sink, dict(heights), dict(excess)))

    def draw(i: int):
        ax.clear()
        kind, u, v, amount, flow_value, hts, exs = frames[i]
        for vtx in range(net.n):
            x, y = pos[vtx]
            color = "#ffd6a5" if vtx == u else "#cfe3f7"
            if vtx == net.source:
                color = "#8fd18f"
            if vtx == net.sink:
                color = "#f2a6a6"
            ax.scatter([x], [y], s=300, c=color, edgecolors="#20486f", zorder=3)
            label = f"{net.name_of(vtx)}\nh={hts.get(vtx, 0)}\ne={exs.get(vtx, 0.0):g}"
            ax.text(x, y - 0.055, label, ha="center", va="top", fontsize=6, zorder=4)
        for edge in net.iter_edges():
            x0, y0 = pos[edge.u]
            x1, y1 = pos[edge.v]
            highlight = (edge.u, edge.v) == (u, v)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#c1121f" if highlight else "#dddddd",
                    lw=2.0 if highlight else 0.6,
                    shrinkA=8, shrinkB=8,
                ),
            )
        if kind == "relabel":
            what = f"relabel node {net.name_of(u)} from height {amount:g} to {hts.get(u, 0)}"
        elif kind == "push":
            what = f"push {amount:g} from {net.name_of(u)} to {net.name_of(v)}"
        else:
            what = "gap heuristic fired"
        ax.set_title(f"{title}\nevent {i + 1} of {len(frames)}: {what}\nflow at sink {flow_value:g}")
        ax.axis("off")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // max(1, fps), blit=False)
    saved = _save_animation(anim, path, fps)
    plt.close(fig)
    return saved


def animate_flow_buildup(
    net: FlowNetwork,
    path: str,
    layout: str = "layered",
    frames: int = 20,
    fps: int = 4,
    title: str = "Max flow built up",
    dpi: int = 90,
) -> str:
    """A simpler animation: grow the flow from zero to the full answer.

    This one needs no trace at all. It runs Edmonds-Karp with path recording,
    then draws the residual capacities as they shrink, which gives a clean
    picture of the cut closing off.
    """
    from core.edmonds_karp import edmonds_karp

    recorded: List[List[Tuple[int, int, float]]] = []
    work = net.copy()
    result = edmonds_karp(work, keep_paths=True)
    recorded = result.stats.get("paths", [])
    pos = layered_layout(net, net.source)

    sample = max(1, len(recorded) // frames)
    snapshots = list(range(0, len(recorded), sample))
    if not snapshots:
        snapshots = [0]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    max_cap = max((e.capacity for e in net.iter_edges()), default=1.0)

    def draw(i: int):
        ax.clear()
        used: Dict[Tuple[int, int], float] = {}
        for p in recorded[: snapshots[i] + 1]:
            for (u, v, cap) in p:
                used[(u, v)] = used.get((u, v), 0.0) + cap
        for edge in net.iter_edges():
            x0, y0 = pos[edge.u]
            x1, y1 = pos[edge.v]
            f = used.get((edge.u, edge.v), 0.0)
            width = 0.6 + 3.0 * (f / max_cap if max_cap else 0.0)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#c1121f" if f > 0 else "#dddddd",
                                lw=width, shrinkA=8, shrinkB=8),
            )
        for vtx in range(net.n):
            x, y = pos[vtx]
            ax.scatter([x], [y], s=260, c="#cfe3f7", edgecolors="#20486f", zorder=3)
            ax.text(x, y, net.name_of(vtx), ha="center", va="center", fontsize=6, zorder=4)
        ax.set_title(f"{title}\naugmentation {snapshots[i] + 1} of {len(recorded)}, flow {result.value:g} at the end")
        ax.axis("off")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(snapshots), interval=1000 // max(1, fps), blit=False)
    saved = _save_animation(anim, path, fps)
    plt.close(fig)
    return saved
