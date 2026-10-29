"""Command line entry point for the flow network solver.

Examples, all run from the project root:

    python main.py --ek --graph examples/sample_network.json --viz out/ek.png
    python main.py --pr --generate sparse --nodes 5000 --seed 5300
    python main.py --mcf --generate dense --nodes 2000 --seed 5301
    python main.py --match --left 12 --right 12 --degree 3 --seed 5300
    python main.py --segment --image examples/sample_image.npy --out out/mask.npy
    python main.py --schedule --jobs 5 --workers 4 --seed 5300

Exactly one solver flag and at most one application flag are expected. A run
prints a short summary, validates the answer, and writes any requested files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import FlowNetwork
from core.push_relabel import push_relabel
from utils.generators import (
    bipartite_matching_graph,
    random_dense,
    random_sparse,
    grid_graph,
)
from utils.validators import summarize_report, validate_flow


def load_graph(path: str) -> FlowNetwork:
    """Read a JSON graph: {"n": int, "source": int, "sink": int, "edges": [[u, v, cap, cost], ...]}.

    cost is optional on each edge. Names are optional too, as {"names": {"0": "s"}}.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    n = int(data["n"])
    net = FlowNetwork(n, int(data["source"]), int(data["sink"]))
    for edge in data["edges"]:
        if len(edge) == 3:
            u, v, cap = edge
            net.add_edge(int(u), int(v), float(cap))
        else:
            u, v, cap, cost = edge
            net.add_edge(int(u), int(v), float(cap), float(cost))
    for key, name in data.get("names", {}).items():
        net.set_name(int(key), name)
    return net


def save_graph(net: FlowNetwork, path: str) -> None:
    """Write a network back out in the format load_graph reads."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = {
        "n": net.n,
        "source": net.source,
        "sink": net.sink,
        "edges": [[e.u, e.v, e.capacity, e.cost] for e in net.iter_edges()],
        "names": {str(k): v for k, v in net._names.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def build_graph(args) -> FlowNetwork:
    """Build the graph from whichever input flag was given."""
    if args.graph:
        return load_graph(args.graph)
    if args.generate == "sparse":
        return random_sparse(args.nodes, args.seed)
    if args.generate == "dense":
        return random_dense(args.nodes, args.seed)
    if args.generate == "grid":
        side = max(2, int(round(args.nodes ** 0.5)))
        return grid_graph(side, side, args.seed)
    raise SystemExit("pass --graph FILE or --generate sparse|dense|grid")


def report_flow(net: FlowNetwork, result, label: str, elapsed: float, check_cost: bool = False) -> None:
    """Print the summary and the validation for a finished run."""
    report = validate_flow(
        net,
        result.flow,
        net.source,
        net.sink,
        result.value,
        claimed_cost=result.stats.get("cost") if check_cost else None,
    )
    print(f"{label} on {net.n} nodes and {net.original_edge_count} edges")
    print(f"  flow value : {result.value:g}")
    print(f"  time       : {elapsed * 1000:.1f} ms")
    print(f"  work count : {result.augmentations} augmentations, {result.iterations} iterations")
    if "cost" in result.stats:
        print(f"  total cost : {result.stats['cost']:g}")
    print("  checks:")
    print(summarize_report(report))
    if not report["ok"]:
        raise SystemExit("validation failed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Flow network solver")
    parser.add_argument("--ek", action="store_true", help="solve max flow with Edmonds-Karp")
    parser.add_argument("--pr", action="store_true", help="solve max flow with push-relabel")
    parser.add_argument("--mcf", action="store_true", help="solve min cost max flow")
    parser.add_argument("--match", action="store_true", help="bipartite matching application")
    parser.add_argument("--segment", action="store_true", help="image segmentation application")
    parser.add_argument("--schedule", action="store_true", help="scheduling application")
    parser.add_argument("--graph", type=str, default=None, help="path to a JSON graph")
    parser.add_argument("--generate", type=str, default=None, choices=["sparse", "dense", "grid"])
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--left", type=int, default=12)
    parser.add_argument("--right", type=int, default=12)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image", type=str, default=None, help=".npy grayscale image")
    parser.add_argument("--out", type=str, default=None, help="output file for a mask or graph")
    parser.add_argument("--viz", type=str, default=None, help="output png for the network")
    parser.add_argument("--animation", type=str, default=None, help="output gif for the run")
    parser.add_argument("--variant", type=str, default="highest_label", choices=["highest_label", "fifo"])
    args = parser.parse_args(argv)

    solvers = sum([args.ek, args.pr, args.mcf])
    apps = sum([args.match, args.segment, args.schedule])
    if solvers + apps == 0:
        parser.print_help()
        return 1
    if solvers + apps > 1 and not (args.match or args.segment or args.schedule):
        raise SystemExit("pick exactly one of --ek, --pr, --mcf")

    if args.match:
        from applications.matching import bipartite_matching, summarize_matching
        from visualization.plot_network import draw_matching

        left, right, edges = bipartite_matching_graph(args.left, args.right, args.degree, args.seed)
        start = time.perf_counter()
        pairs = bipartite_matching(left, right, edges)
        elapsed = time.perf_counter() - start
        print(f"bipartite matching, {left} left and {right} right, {len(edges)} allowed pairs")
        print(f"  matching size: {len(pairs)}")
        print(f"  time         : {elapsed * 1000:.1f} ms")
        print(summarize_matching(pairs))
        if args.viz:
            print("  figure       :", draw_matching(left, right, edges, pairs, args.viz))
        return 0

    if args.segment:
        from applications.image_seg import make_test_image, mask_accuracy, segment_image
        from visualization.plot_network import draw_segmentation

        if args.image:
            image = np.load(args.image)
        else:
            image = make_test_image(64, args.seed)
            print("no --image given, using the built in test image")
        start = time.perf_counter()
        result = segment_image(image, solver="mcf" if args.mcf else "pr")
        elapsed = time.perf_counter() - start
        truth = image > 0.45
        print(f"segmentation on a {image.shape[0]} by {image.shape[1]} image, {result['nodes']} nodes")
        print(f"  cut capacity : {result['cut_capacity']:g}")
        print(f"  accuracy     : {mask_accuracy(result['mask'], truth):.4f}")
        print(f"  time         : {elapsed * 1000:.1f} ms")
        if args.out:
            np.save(args.out, result["mask"])
            print("  mask saved   :", args.out)
        if args.viz:
            print("  figure       :", draw_segmentation(image, result["mask"], args.viz))
        return 0

    if args.schedule:
        from applications.scheduling import format_schedule, solve_assignment, staffing_shortfall

        rng = np.random.default_rng(args.seed)
        demands = [int(x) for x in rng.integers(2, 9, size=args.jobs)]
        capacity = [int(x) for x in rng.integers(6, 16, size=args.workers)]
        allowed = []
        for j in range(args.jobs):
            for w in range(args.workers):
                if rng.random() < 0.6:
                    allowed.append((j, w))
        if not allowed:
            allowed = [(j, w) for j in range(args.jobs) for w in range(args.workers)]
        result = solve_assignment(demands, capacity, allowed)
        print(f"scheduling {args.jobs} jobs on {args.workers} workers, seed {args.seed}")
        print(f"  demand   : {result['demand']:g} hours")
        print(f"  assigned : {result['assigned']:g} hours")
        print(f"  feasible : {result['feasible']}")
        print("  schedule :")
        print(format_schedule(result))
        short = staffing_shortfall(demands, capacity, allowed)
        if short:
            print("  short jobs:", ", ".join(f"job {j} needs {h:g} more" for j, h in short))
        return 0

    net = build_graph(args)
    if args.ek:
        start = time.perf_counter()
        result = edmonds_karp(net, keep_paths=bool(args.animation))
        elapsed = time.perf_counter() - start
        report_flow(net, result, "Edmonds-Karp", elapsed)
        if args.viz:
            from visualization.plot_network import draw_cut, draw_network

            print("  figure       :", draw_network(net, args.viz, result.flow, title="Edmonds-Karp"))
        if args.animation:
            from visualization.animate_flow import animate_edmonds_karp

            print("  animation    :", animate_edmonds_karp(net, result.stats["paths"], args.animation))
        return 0

    if args.pr:
        start = time.perf_counter()
        result = push_relabel(net, variant=args.variant, trace=bool(args.animation))
        elapsed = time.perf_counter() - start
        report_flow(net, result, f"Push-relabel ({args.variant})", elapsed)
        print(f"  pushes     : {result.stats['pushes']}, relabels {result.stats['relabels']}")
        if args.viz:
            from visualization.plot_network import draw_network

            print("  figure       :", draw_network(net, args.viz, result.flow, title="Push-relabel"))
        if args.animation:
            from visualization.animate_flow import animate_push_relabel

            print("  animation    :", animate_push_relabel(net, result.stats["trace"], args.animation))
        return 0

    if args.mcf:
        start = time.perf_counter()
        result = min_cost_flow(net)
        elapsed = time.perf_counter() - start
        report_flow(net, result, "Min cost max flow", elapsed, check_cost=True)
        if args.viz:
            from visualization.plot_network import draw_network

            print("  figure       :", draw_network(net, args.viz, result.flow, title="Min cost max flow"))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
