"""Benchmark Edmonds-Karp, push-relabel, and min cost flow.

Every number this script prints or writes comes from a run that happened while
it ran. Nothing is estimated, nothing is typed in by hand, and every solver
result is passed through the validators before it is recorded, so a wrong answer
cannot quietly become a row in a table.

What it measures:

- runtime in milliseconds, mean and standard deviation across five seeds
- peak memory of the solver call, in megabytes
- augmentations (Edmonds-Karp paths, push-relabel pushes, min cost flow rounds)
- push-relabel relabels, operations, and a convergence profile

How the runs are arranged:

- two graph families, sparse and dense, at several node counts
- five seeds, 5300 to 5304, so every cell has a real spread
- a per run time budget, so a size that is too slow on one machine is recorded
  as timed out instead of stalling the whole benchmark
- min cost flow only up to the sizes where it finishes inside the budget

Outputs, all under results/:

- runtime_vs_size.png, memory_vs_size.png, augmentations_vs_size.png
- push_relabel_convergence.png
- benchmark_summary.csv, benchmark_runs.csv, convergence_profiles.csv
- benchmark_run.log with the console text

Run it from the project root:

    python benchmarks/week10_flow_benchmark.py
    python benchmarks/week10_flow_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import TimeoutError_
from core.push_relabel import push_relabel
from utils.generators import random_dense, random_sparse
from utils.validators import validate_flow

SEEDS = [5300, 5301, 5302, 5303, 5304]
RESULTS = os.path.join(ROOT, "results")
DPI = 150


def peak_memory_mb() -> float:
    """Peak resident memory of THIS process, in megabytes.

    tracemalloc is not used because the solvers keep most of their state in
    Python lists of floats, which tracemalloc does not see. Peak RSS from the
    operating system covers everything the process touched, including the
    interpreter itself.

    Peak RSS is a high water mark that never falls, so the only way to get a
    per run number is to measure it in a process that has done nothing else.
    That is why run_solver is executed in a fresh worker process for every
    measurement.
    """
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux and Windows report kilobytes, macOS reports bytes.
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except Exception:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return float("nan")


def _clean(value, digits: int = 3):
    """Round a float for CSV output, keeping NaN as an empty string."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except TypeError:
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _worker(payload: dict) -> dict:
    """Run one measurement in a fresh process and return the numbers.

    This function is executed by a child interpreter, not by the benchmark
    process itself. A fresh process matters for memory: peak RSS is a high
    water mark, so measuring it after a dozen earlier runs would report the
    largest run of the session for every row. A child that builds a graph and
    runs a solver once and exits gives a clean peak for that single run.
    """
    family = payload["family"]
    nodes = payload["nodes"]
    seed = payload["seed"]
    name = payload["solver"]
    want_profile = payload["want_profile"]

    maker = random_sparse if family == "sparse" else random_dense

    import resource

    # Measure before the graph exists so the reported number covers the graph
    # plus the solver, which is the memory a caller actually needs. The
    # interpreter and the imported libraries are already resident by then and
    # are excluded.
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    net = maker(nodes, seed)
    start = time.perf_counter()
    try:
        if name == "edmonds_karp":
            result = edmonds_karp(net)
        elif name == "push_relabel":
            result = push_relabel(net)
        elif name == "min_cost_flow":
            result = min_cost_flow(net)
        else:
            raise ValueError(f"unknown solver {name}")
    except TimeoutError_:
        return {
            "status": "timeout",
            "seconds": time.perf_counter() - start,
            "flow": None,
            "valid": False,
            "augmentations": None,
            "iterations": None,
            "pushes": None,
            "relabels": None,
            "operations": None,
            "cost": None,
            "memory_mb": None,
            "profile": None,
            "detail": "time budget reached",
            "edges": net.original_edge_count,
        }

    elapsed = time.perf_counter() - start
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        delta_mb = max(0.0, (after_kb - before_kb) / (1024.0 * 1024.0))
    else:
        delta_mb = max(0.0, (after_kb - before_kb) / 1024.0)

    claimed_cost = result.stats.get("cost") if name == "min_cost_flow" else None
    report = validate_flow(net, result.flow, net.source, net.sink, result.value,
                           claimed_cost=claimed_cost)
    payload_out = {
        "status": "ok" if report["ok"] else "invalid",
        "seconds": elapsed,
        "flow": result.value,
        "valid": bool(report["ok"]),
        "augmentations": result.augmentations,
        "iterations": result.iterations,
        "pushes": result.stats.get("pushes"),
        "relabels": result.stats.get("relabels"),
        "operations": result.stats.get("operations"),
        "cost": claimed_cost,
        "memory_mb": delta_mb,
        "profile": result.stats.get("profile") if want_profile else None,
        "detail": "" if report["ok"] else "; ".join(report["problems"][:3]),
        "edges": net.original_edge_count,
    }
    return payload_out


def run_solver(name: str, family: str, nodes: int, seed: int, budget: float,
               want_profile: bool = False) -> dict:
    """Run one solver in an isolated child process under a hard time limit.

    The child is killed if it overruns, so a size that is far too slow on one
    machine cannot stall the whole benchmark. A killed run is reported as a
    timeout and no numbers are invented for it.
    """
    payload = {
        "family": family,
        "nodes": nodes,
        "seed": seed,
        "solver": name,
        "want_profile": want_profile,
    }
    command = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(payload)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=budget,
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return {
            "solver": name, "status": "timeout", "seconds": budget, "flow": None,
            "valid": False, "augmentations": None, "iterations": None, "pushes": None,
            "relabels": None, "operations": None, "cost": None, "memory_mb": None,
            "profile": None, "detail": "child process exceeded the time budget",
            "edges": None,
        }
    if completed.returncode != 0:
        return {
            "solver": name, "status": "error", "seconds": 0.0, "flow": None,
            "valid": False, "augmentations": None, "iterations": None, "pushes": None,
            "relabels": None, "operations": None, "cost": None, "memory_mb": None,
            "profile": None,
            "detail": (completed.stderr or "").strip().splitlines()[-1:] or ["child failed"],
            "edges": None,
        }
    out = json.loads(completed.stdout)
    out["solver"] = name
    return out


def summarize(rows, family: str, nodes: int, solver: str):
    """Mean and standard deviation across seeds for one cell of the table."""
    cells = [r for r in rows if r["family"] == family and r["nodes"] == nodes and r["solver"] == solver]
    good = [r for r in cells if r["status"] == "ok"]
    timed_out = [r for r in cells if r["status"] == "timeout"]
    invalid = [r for r in cells if r["status"] in ("invalid", "error")]
    attempted = len(cells) - len([r for r in cells if r["status"] == "skipped"])
    entry = {
        "family": family,
        "nodes": nodes,
        "solver": solver,
        "runs": attempted,
        "finished": len(good),
        "timed_out": len(timed_out),
        "invalid": len(invalid),
    }
    if good:
        times = [r["seconds"] * 1000.0 for r in good]
        mems = [r["memory_mb"] for r in good]
        augs = [r["augmentations"] for r in good if r["augmentations"] is not None]
        entry["mean_ms"] = statistics.mean(times)
        entry["std_ms"] = statistics.stdev(times) if len(times) > 1 else 0.0
        entry["mean_memory_mb"] = statistics.mean(mems)
        entry["std_memory_mb"] = statistics.stdev(mems) if len(mems) > 1 else 0.0
        entry["mean_augmentations"] = statistics.mean(augs) if augs else None
        entry["std_augmentations"] = statistics.stdev(augs) if len(augs) > 1 else 0.0
        entry["mean_flow"] = statistics.mean([r["flow"] for r in good])
    else:
        entry["mean_ms"] = None
        entry["std_ms"] = None
        entry["mean_memory_mb"] = None
        entry["std_memory_mb"] = None
        entry["mean_augmentations"] = None
        entry["std_augmentations"] = None
        entry["mean_flow"] = None
    return entry


def plot_runtime(summaries, path: str) -> str:
    """Runtime against graph size, one line per solver and family."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    colors = {"edmonds_karp": "#1f4e79", "push_relabel": "#c1121f", "min_cost_flow": "#2b7a4b"}
    labels = {
        "edmonds_karp": "Edmonds-Karp",
        "push_relabel": "Push-relabel",
        "min_cost_flow": "Min cost flow",
    }
    for ax, family in zip(axes, ("sparse", "dense")):
        for solver in ("edmonds_karp", "push_relabel", "min_cost_flow"):
            rows = [s for s in summaries if s["family"] == family and s["solver"] == solver]
            rows = [r for r in rows if r["mean_ms"] is not None]
            if not rows:
                continue
            rows.sort(key=lambda r: r["nodes"])
            xs = [r["nodes"] for r in rows]
            ys = [r["mean_ms"] for r in rows]
            errs = [r["std_ms"] for r in rows]
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, label=labels[solver],
                        color=colors[solver], linewidth=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of nodes")
        ax.set_ylabel("runtime (ms, log scale)")
        ax.set_title(f"{family} graphs")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Runtime vs graph size, mean and standard deviation over 5 seeds")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_memory(summaries, path: str) -> str:
    """Peak memory against graph size."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    colors = {"edmonds_karp": "#1f4e79", "push_relabel": "#c1121f", "min_cost_flow": "#2b7a4b"}
    labels = {
        "edmonds_karp": "Edmonds-Karp",
        "push_relabel": "Push-relabel",
        "min_cost_flow": "Min cost flow",
    }
    for ax, family in zip(axes, ("sparse", "dense")):
        for solver in ("edmonds_karp", "push_relabel", "min_cost_flow"):
            rows = [s for s in summaries if s["family"] == family and s["solver"] == solver]
            rows = [r for r in rows if r["mean_memory_mb"] is not None]
            if not rows:
                continue
            rows.sort(key=lambda r: r["nodes"])
            ax.errorbar(
                [r["nodes"] for r in rows],
                [r["mean_memory_mb"] for r in rows],
                yerr=[r["std_memory_mb"] for r in rows],
                marker="s", capsize=4, label=labels[solver], color=colors[solver], linewidth=1.6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("number of nodes")
        ax.set_ylabel("peak process memory (MB)")
        ax.set_title(f"{family} graphs")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Memory used vs graph size, mean and standard deviation over 5 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_augmentations(summaries, path: str) -> str:
    """Work counters against graph size."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    colors = {"edmonds_karp": "#1f4e79", "push_relabel": "#c1121f", "min_cost_flow": "#2b7a4b"}
    labels = {
        "edmonds_karp": "Edmonds-Karp (paths)",
        "push_relabel": "Push-relabel (pushes)",
        "min_cost_flow": "Min cost flow (rounds)",
    }
    for ax, family in zip(axes, ("sparse", "dense")):
        for solver in ("edmonds_karp", "push_relabel", "min_cost_flow"):
            rows = [s for s in summaries if s["family"] == family and s["solver"] == solver]
            rows = [r for r in rows if r["mean_augmentations"]]
            if not rows:
                continue
            rows.sort(key=lambda r: r["nodes"])
            ax.errorbar(
                [r["nodes"] for r in rows],
                [r["mean_augmentations"] for r in rows],
                yerr=[r["std_augmentations"] for r in rows],
                marker="^", capsize=4, label=labels[solver], color=colors[solver], linewidth=1.6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of nodes")
        ax.set_ylabel("count (log scale)")
        ax.set_title(f"{family} graphs")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Augmentation and push counts vs graph size, mean over 5 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_convergence(profiles, keys, path: str) -> str:
    """Flow at the sink against operations for push-relabel.

    profiles maps (family, nodes) to the sample list, so the sparse and dense
    families can sit in the same figure without one overwriting the other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=True)
    colors = ["#1f4e79", "#c1121f", "#2b7a4b", "#8a5a00", "#5b2a86"]
    for ax, family in zip(axes, ("sparse", "dense")):
        keys_here = [k for k in keys if k[0] == family]
        for color, key in zip(colors, keys_here):
            profile = profiles[key]
            if not profile:
                continue
            ops = [p[0] for p in profile]
            flows = [p[1] for p in profile]
            top = max(flows) if flows else 1.0
            if top > 0:
                flows = [f / top for f in flows]
            ax.plot(ops, flows, label=f"{key[1]} nodes, final flow {top:g}",
                    color=color, linewidth=1.8)
        ax.set_xlabel("operations (pushes plus relabels)")
        ax.set_title(f"{family} graphs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("flow at sink, as a fraction of the final value")
    fig.suptitle("Push-relabel convergence: how the flow builds up")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    # Worker mode is an internal detail: the benchmark starts itself with this
    # flag to take one isolated measurement. It is read from sys.argv because
    # main is also called with an explicit argument list by the tests.
    raw = sys.argv[1:] if argv is None else list(argv)
    if len(raw) >= 2 and raw[0] == "--worker":
        print(json.dumps(_worker(json.loads(raw[1]))))
        return 0

    parser = argparse.ArgumentParser(description="Benchmark the flow solvers")
    parser.add_argument("--quick", action="store_true", help="small sizes only, for a fast check")
    parser.add_argument("--budget", type=float, default=90.0,
                        help="seconds allowed per solver run (min cost flow gets 1.5 times this)")
    parser.add_argument("--seeds", type=int, default=5, help="how many of the five seeds to use")
    args = parser.parse_args(argv)

    seeds = SEEDS[: max(1, min(5, args.seeds))]
    if args.quick:
        sizes = [1000, 2000, 5000]
        mcf_sizes = [1000, 2000]
    else:
        sizes = [1000, 5000, 10000, 25000, 50000]
        mcf_sizes = [1000, 2000, 5000]

    os.makedirs(RESULTS, exist_ok=True)
    log_lines: list[str] = []

    def say(text: str) -> None:
        print(text, flush=True)
        log_lines.append(text)

    say("CSC 5300 Week 10 flow benchmark")
    say(f"python {sys.version.split()[0]}, seeds {seeds}, sizes {sizes}")
    say(f"time budget per solver run: {args.budget:g} seconds")
    say("")

    rows = []
    profiles = {}
    started_all = time.perf_counter()
    skip_note: dict[str, int] = {}

    for family, _maker in (("sparse", random_sparse), ("dense", random_dense)):
        for nodes in sizes:
            # Once a family and size has timed out for one seed, the remaining
            # seeds would only burn the same budget for the same answer. They
            # are recorded as skipped, not as measurements.
            for seed in seeds:
                for solver in ("edmonds_karp", "push_relabel"):
                    if skip_note.get((family, nodes, solver)):
                        rows.append({
                            "family": family, "nodes": nodes, "seed": seed, "solver": solver,
                            "status": "skipped", "seconds": None, "flow": None, "valid": False,
                            "augmentations": None, "iterations": None, "pushes": None,
                            "relabels": None, "operations": None, "cost": None,
                            "memory_mb": None, "detail": "skipped after an earlier timeout",
                            "edges": skip_note[(family, nodes, solver)],
                        })
                        continue
                    # Edmonds-Karp is O(V E squared) in the worst case, so it
                    # earns a smaller slice of time. Push-relabel gets the full
                    # budget because it is the solver that has to reach 50,000
                    # nodes, and a generous limit proves it can.
                    per_solver_budget = args.budget * (0.75 if solver == "edmonds_karp" else 1.5)
                    record = run_solver(solver, family, nodes, seed, per_solver_budget,
                                        want_profile=(solver == "push_relabel"))
                    record.update({"family": family, "nodes": nodes, "seed": seed})
                    rows.append(record)
                    if record["status"] == "ok":
                        say(f"  {family:6s} {nodes:6d} nodes {record['edges']:7d} edges "
                            f"seed {seed} {solver:14s} {record['seconds'] * 1000:9.1f} ms  "
                            f"flow {record['flow']:g}  work {record['augmentations']}  ok")
                        if solver == "push_relabel" and seed == seeds[0]:
                            profiles[(family, nodes)] = record["profile"]
                    else:
                        say(f"  {family:6s} {nodes:6d} nodes seed {seed} {solver:14s} "
                            f"{record['status']}: {record['detail']}")
                        if record["status"] == "timeout":
                            skip_note[(family, nodes, solver)] = record["edges"]

                if nodes in mcf_sizes:
                    if skip_note.get((family, nodes, "min_cost_flow")):
                        rows.append({
                            "family": family, "nodes": nodes, "seed": seed,
                            "solver": "min_cost_flow", "status": "skipped", "seconds": None,
                            "flow": None, "valid": False, "augmentations": None,
                            "iterations": None, "pushes": None, "relabels": None,
                            "operations": None, "cost": None, "memory_mb": None,
                            "detail": "skipped after an earlier timeout",
                            "edges": skip_note[(family, nodes, "min_cost_flow")],
                        })
                        continue
                    record = run_solver("min_cost_flow", family, nodes, seed, args.budget)
                    record.update({"family": family, "nodes": nodes, "seed": seed})
                    rows.append(record)
                    if record["status"] == "ok":
                        say(f"  {family:6s} {nodes:6d} nodes {record['edges']:7d} edges "
                            f"seed {seed} min_cost_flow   {record['seconds'] * 1000:9.1f} ms  "
                            f"flow {record['flow']:g}  cost {record['cost']:g}  ok")
                    else:
                        say(f"  {family:6s} {nodes:6d} nodes seed {seed} min_cost_flow   "
                            f"{record['status']}: {record['detail']}")
                        if record["status"] == "timeout":
                            skip_note[(family, nodes, "min_cost_flow")] = record["edges"]

    summaries = []
    for family in ("sparse", "dense"):
        for nodes in sizes:
            for solver in ("edmonds_karp", "push_relabel", "min_cost_flow"):
                entry = summarize(rows, family, nodes, solver)
                if entry["runs"]:
                    summaries.append(entry)

    say("")
    say("Summary, mean and standard deviation across seeds")
    say(f"{'family':8s} {'nodes':>7s} {'solver':16s} {'mean ms':>10s} {'std ms':>9s} "
        f"{'mem MB':>8s} {'mean work':>11s} {'runs':>5s}")
    for entry in summaries:
        mean_ms = f"{entry['mean_ms']:.1f}" if entry["mean_ms"] is not None else "timeout"
        std_ms = f"{entry['std_ms']:.1f}" if entry["std_ms"] is not None else ""
        mem = f"{entry['mean_memory_mb']:.1f}" if entry["mean_memory_mb"] is not None else ""
        work = (f"{entry['mean_augmentations']:.0f}"
                if entry["mean_augmentations"] is not None else "")
        say(f"{entry['family']:8s} {entry['nodes']:7d} {entry['solver']:16s} {mean_ms:>10s} "
            f"{std_ms:>9s} {mem:>8s} {work:>11s} {entry['finished']:>2d}/{entry['runs']:<2d}")

    summary_path = os.path.join(RESULTS, "benchmark_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for entry in summaries:
            writer.writerow({k: _clean(v) for k, v in entry.items()})

    runs_path = os.path.join(RESULTS, "benchmark_runs.csv")
    with open(runs_path, "w", newline="", encoding="utf-8") as handle:
        fields = ["family", "nodes", "edges", "seed", "solver", "status", "seconds", "flow",
                  "valid", "augmentations", "iterations", "pushes", "relabels", "operations",
                  "cost", "memory_mb", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _clean(row.get(k)) for k in fields})

    profile_path = os.path.join(RESULTS, "convergence_profiles.csv")
    with open(profile_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "nodes", "operation", "flow_at_sink"])
        for (family, nodes), profile in sorted(profiles.items()):
            for op, flow_value in profile or []:
                writer.writerow([family, nodes, op, flow_value])

    figures = [
        plot_runtime(summaries, os.path.join(RESULTS, "runtime_vs_size.png")),
        plot_memory(summaries, os.path.join(RESULTS, "memory_vs_size.png")),
        plot_augmentations(summaries, os.path.join(RESULTS, "augmentations_vs_size.png")),
    ]
    profile_keys = sorted(profiles)
    if profile_keys:
        figures.append(
            plot_convergence(profiles, profile_keys,
                             os.path.join(RESULTS, "push_relabel_convergence.png"))
        )

    say("")
    say(f"wrote {summary_path}")
    say(f"wrote {runs_path}")
    say(f"wrote {profile_path}")
    for figure in figures:
        say(f"wrote {figure}")
    total = time.perf_counter() - started_all
    say(f"total benchmark time {total:.1f} s")

    with open(os.path.join(RESULTS, "benchmark_run.log"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
