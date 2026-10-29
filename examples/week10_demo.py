"""A short guided tour of everything in the project.

Run it from the project root:

    python examples/week10_demo.py

Each section builds a tiny instance, runs an algorithm, and prints what came
out. The point is a readable check that each part works, not a benchmark.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from applications.image_seg import make_test_image, mask_accuracy, segment_image
from applications.matching import (
    bipartite_matching,
    brute_force_max_matching,
    maximum_weight_matching,
)
from applications.scheduling import solve_assignment, solve_slots, tournament_bracket
from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import FlowNetwork
from core.push_relabel import push_relabel
from utils.generators import random_sparse
from utils.validators import summarize_report, validate_flow


def rule(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_small_network() -> None:
    """The same graph solved three ways, with a validation of each answer."""
    rule("1. One small network, three solvers")

    net = FlowNetwork(6, 0, 5)
    net.add_edge(0, 1, 16, 1)
    net.add_edge(0, 2, 13, 2)
    net.add_edge(1, 2, 10, 1)
    net.add_edge(2, 1, 4, 1)
    net.add_edge(1, 3, 12, 3)
    net.add_edge(3, 2, 9, 1)
    net.add_edge(2, 4, 14, 2)
    net.add_edge(4, 3, 7, 1)
    net.add_edge(3, 5, 20, 1)
    net.add_edge(4, 5, 4, 1)

    print(f"network: {net.n} nodes, {net.original_edge_count} edges")

    work = net.copy()
    ek = edmonds_karp(work)
    print(f"  Edmonds-Karp     flow {ek.value:g} in {ek.augmentations} augmentations")
    print(f"                   mean path length {ek.stats['mean_path_length']:.2f} edges")

    work = net.copy()
    pr = push_relabel(work)
    print(f"  Push-relabel     flow {pr.value:g} with {pr.stats['pushes']} pushes and "
          f"{pr.stats['relabels']} relabels")

    work = net.copy()
    mcf = min_cost_flow(work)
    print(f"  Min cost flow    flow {mcf.value:g} at total cost {mcf.stats['cost']:g}")

    work = net.copy()
    ek2 = edmonds_karp(work)
    report = validate_flow(work, ek2.flow, 0, 5, ek2.value)
    print("  validation of the Edmonds-Karp answer:")
    print(summarize_report(report))

    work = net.copy()
    ek3 = edmonds_karp(work)
    left, right, cut = work.min_cut(0, ek3.flow)
    print(f"  min cut: source side {sorted(left)}")
    print(f"           sink side   {sorted(right)}")
    print(f"           cut capacity {cut:g} equals the flow, as the theorem says")


def demo_matching() -> None:
    """Bipartite matching, then weighted matching on the same graph."""
    rule("2. Bipartite matching as max flow")

    left = ["Ana", "Ben", "Cara", "Dev", "Eli"]
    right = ["shift A", "shift B", "shift C", "shift D"]
    allowed = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 3), (3, 2), (3, 3), (4, 1)]

    pairs = bipartite_matching(len(left), len(right), allowed)
    best = brute_force_max_matching(len(left), len(right), allowed)
    print(f"  {len(left)} people, {len(right)} shifts, {len(allowed)} allowed pairs")
    for u, v in pairs:
        print(f"    {left[u]} takes {right[v]}")
    print(f"  matching size {len(pairs)}, brute force says {best}")

    weights = {(0, 0): 5, (0, 1): 3, (1, 1): 6, (1, 2): 2, (2, 0): 4, (2, 3): 7, (3, 2): 8, (3, 3): 1, (4, 1): 9}
    best_pairs, total = maximum_weight_matching(len(left), len(right), allowed, weights)
    print(f"  weighted version picks {len(best_pairs)} pairs worth {total:g}")
    for u, v in best_pairs:
        print(f"    {left[u]} takes {right[v]} (weight {weights[(u, v)]})")


def demo_segmentation() -> None:
    """Segment a synthetic image and report accuracy against the truth."""
    rule("3. Image segmentation by graph cut")

    image = make_test_image(48, seed=5300)
    truth = image > 0.45
    result = segment_image(image, solver="pr")
    accuracy = mask_accuracy(result["mask"], truth)
    print(f"  image {image.shape[0]} by {image.shape[1]}, {result['nodes']} nodes, "
          f"{result['edges']} edges")
    print(f"  foreground looks like {result['foreground_mean']:.2f} brightness")
    print(f"  background looks like {result['background_mean']:.2f} brightness")
    print(f"  cut capacity {result['cut_capacity']:.2f}")
    print(f"  accuracy against the known rectangle {accuracy:.4f}")
    print(f"  foreground pixels: {int(result['mask'].sum())} of {result['mask'].size}")


def demo_scheduling() -> None:
    """Assignment with capacities, then slots, then a tournament bracket."""
    rule("4. Scheduling and resource allocation")

    demands = [6, 4, 5, 3]
    capacity = [8, 7, 6]
    allowed = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2), (3, 0), (3, 1), (3, 2)]
    result = solve_assignment(demands, capacity, allowed)
    print(f"  hours demanded {result['demand']:g}, hours assigned {result['assigned']:g}")
    print(f"  feasible: {result['feasible']}")
    for j, w, hours in result["schedule"]:
        print(f"    job {j} gets {hours:g} hours from worker {w}")

    tight = solve_assignment([9, 9], [10, 5], [(0, 0), (1, 1)])
    print(f"  a tight instance: assigned {tight['assigned']:g} of {tight['demand']:g}, "
          f"short by {tight['unmet']:g}")

    slots = solve_slots(4, 3, [(0, 0), (1, 0), (1, 1), (2, 2), (3, 1), (3, 2)])
    print(f"  slot placement: {slots['count']} of {slots['jobs']} jobs placed")
    for j, k in slots["placed"]:
        print(f"    job {j} in slot {k}")

    bracket = tournament_bracket(4, 4, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)])
    print(f"  bracket filled {bracket['filled']} of {bracket['slots']} slots, "
          f"complete: {bracket['complete']}")
    for slot in sorted(bracket["bracket"]):
        print(f"    slot {slot}: team {bracket['bracket'][slot]}")


def demo_larger_graph() -> None:
    """A generated graph at a size that shows the solvers doing real work."""
    rule("5. A generated graph at 2,000 nodes")

    net = random_sparse(2000, seed=5300)
    work = net.copy()
    ek = edmonds_karp(work)
    work = net.copy()
    pr = push_relabel(work)
    print(f"  graph: {net.n} nodes, {net.original_edge_count} edges")
    print(f"  Edmonds-Karp  flow {ek.value:g}, {ek.augmentations} augmentations")
    print(f"  Push-relabel  flow {pr.value:g}, {pr.stats['pushes']} pushes, "
          f"{pr.stats['relabels']} relabels")
    print(f"  both agree: {abs(ek.value - pr.value) < 1e-6}")


def main() -> int:
    print("CSC 5300 Week 10 demo: flow networks, matching, and graph cuts")
    demo_small_network()
    demo_matching()
    demo_segmentation()
    demo_scheduling()
    demo_larger_graph()
    print()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
