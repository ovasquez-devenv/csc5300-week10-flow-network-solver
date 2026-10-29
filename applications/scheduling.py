"""Scheduling and resource allocation as a flow problem.

Two related models live here.

Assignment with capacities. Each job needs some number of hours from one worker,
or each machine can take a limited number of jobs. Source to each job with the
job's demand, job to every worker qualified for it with capacity equal to what
that worker can give, worker to sink with the worker's capacity. The max flow is
the amount of demand that can be met, and the flow on the job to worker edges is
the schedule. If that max flow equals total demand, the instance is feasible
and the flow is a valid assignment.

Time slot scheduling. The same model with jobs on the left, time slots on the
right, and each job allowed in the slots it fits.

Tournament brackets are the same shape: teams on the left, bracket slots on the
right, one team per slot. A flow of size equal to the number of slots is a
complete bracket.

Costs are supported so a schedule can prefer the cheapest worker or the earliest
slot, using min cost flow with all demands met.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.edmonds_karp import edmonds_karp
from core.min_cost_flow import min_cost_flow
from core.network import FlowNetwork


def build_assignment_network(
    job_demands: Sequence[float],
    worker_capacity: Sequence[float],
    allowed: Iterable[Tuple[int, int]],
    costs: Optional[Dict[Tuple[int, int], float]] = None,
) -> Tuple[FlowNetwork, int, int]:
    """Flow network for assigning jobs to workers.

    Node layout: source 0, jobs 1..J, workers 1 + J..J + W, sink last.
    allowed holds (job, worker) pairs that are permitted. costs holds optional
    per pair costs, where a higher cost is worse.
    """
    jobs = len(job_demands)
    workers = len(worker_capacity)
    source = 0
    job_base = 1
    worker_base = 1 + jobs
    sink = worker_base + workers
    net = FlowNetwork(sink + 1, source, sink)

    for j, demand in enumerate(job_demands):
        if demand < 0:
            raise ValueError("job demand must be non negative")
        net.add_edge(source, job_base + j, float(demand))
    for w, cap in enumerate(worker_capacity):
        if cap < 0:
            raise ValueError("worker capacity must be non negative")
        net.add_edge(worker_base + w, sink, float(cap))
    for j, w in allowed:
        if not (0 <= j < jobs and 0 <= w < workers):
            raise IndexError(f"pair ({j}, {w}) outside the instance")
        cost = 0.0 if costs is None else float(costs.get((j, w), 0.0))
        # Capacity is the smaller of what the job needs and what the worker has.
        cap = min(float(job_demands[j]), float(worker_capacity[w]))
        net.add_edge(job_base + j, worker_base + w, cap, cost)
    return net, source, sink


def solve_assignment(
    job_demands: Sequence[float],
    worker_capacity: Sequence[float],
    allowed: Iterable[Tuple[int, int]],
    costs: Optional[Dict[Tuple[int, int], float]] = None,
    solver: str = "ek",
) -> Dict[str, object]:
    """Assign work to workers and report feasibility.

    Returns a dict with the schedule, total assigned hours, demand, the unmet
    amount, whether the instance is feasible, and the cost when costs are given.
    """
    allowed = list(allowed)
    net, s, t = build_assignment_network(job_demands, worker_capacity, allowed, costs)
    if costs is None:
        result = edmonds_karp(net, s, t) if solver == "ek" else None
        if result is None:
            from core.push_relabel import push_relabel

            result = push_relabel(net, s, t)
        cost = 0.0
    else:
        result = min_cost_flow(net, s, t)
        cost = result.stats["cost"]

    jobs = len(job_demands)
    job_base = 1
    worker_base = 1 + jobs
    schedule: List[Tuple[int, int, float]] = []
    per_job: Dict[int, float] = {j: 0.0 for j in range(jobs)}
    per_worker: Dict[int, float] = {w: 0.0 for w in range(len(worker_capacity))}
    for (u, v), f in result.flow.items():
        if f <= 1e-12:
            continue
        if job_base <= u < job_base + jobs and worker_base <= v < worker_base + len(worker_capacity):
            j = u - job_base
            w = v - worker_base
            schedule.append((j, w, f))
            per_job[j] += f
            per_worker[w] += f
    schedule.sort()

    total_demand = float(sum(job_demands))
    return {
        "schedule": schedule,
        "assigned": result.value,
        "demand": total_demand,
        "unmet": max(0.0, total_demand - result.value),
        "feasible": abs(total_demand - result.value) < 1e-9,
        "cost": cost,
        "per_job": per_job,
        "per_worker": per_worker,
        "nodes": net.n,
        "edges": net.original_edge_count,
    }


def build_slot_network(
    n_jobs: int,
    n_slots: int,
    available: Iterable[Tuple[int, int]],
    slot_capacity: Optional[Sequence[float]] = None,
) -> Tuple[FlowNetwork, int, int]:
    """Flow network for putting jobs into time slots, one unit per job."""
    source = 0
    job_base = 1
    slot_base = 1 + n_jobs
    sink = slot_base + n_slots
    net = FlowNetwork(sink + 1, source, sink)
    for j in range(n_jobs):
        net.add_edge(source, job_base + j, 1.0)
    caps = slot_capacity if slot_capacity is not None else [1.0] * n_slots
    for k, cap in enumerate(caps):
        net.add_edge(slot_base + k, sink, float(cap))
    for j, k in available:
        if not (0 <= j < n_jobs and 0 <= k < n_slots):
            raise IndexError(f"pair ({j}, {k}) outside the instance")
        net.add_edge(job_base + j, slot_base + k, 1.0)
    return net, source, sink


def solve_slots(
    n_jobs: int,
    n_slots: int,
    available: Iterable[Tuple[int, int]],
    slot_capacity: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Place as many jobs into slots as possible. Returns the slot assignment."""
    net, s, t = build_slot_network(n_jobs, n_slots, available, slot_capacity)
    result = edmonds_karp(net, s, t)

    job_base = 1
    slot_base = 1 + n_jobs
    placed: List[Tuple[int, int]] = []
    for (u, v), f in result.flow.items():
        if f <= 1e-12:
            continue
        if job_base <= u < job_base + n_jobs and slot_base <= v < slot_base + n_slots:
            placed.append((u - job_base, v - slot_base))
    placed.sort()
    return {
        "placed": placed,
        "count": len(placed),
        "jobs": n_jobs,
        "all_placed": len(placed) == n_jobs,
        "nodes": net.n,
        "edges": net.original_edge_count,
    }


def tournament_bracket(
    n_teams: int,
    n_slots: int,
    eligible: Iterable[Tuple[int, int]],
) -> Dict[str, object]:
    """Fill bracket slots with teams, one team per slot.

    eligible holds (team, slot) pairs where that team may appear. A perfect
    bracket exists exactly when the max flow equals the number of slots.
    """
    net, s, t = build_slot_network(n_teams, n_slots, eligible)
    result = edmonds_karp(net, s, t)
    team_base = 1
    slot_base = 1 + n_teams
    bracket: Dict[int, int] = {}
    for (u, v), f in result.flow.items():
        if f <= 1e-12:
            continue
        if team_base <= u < team_base + n_teams and slot_base <= v < slot_base + n_slots:
            bracket[v - slot_base] = u - team_base
    return {
        "bracket": bracket,
        "filled": len(bracket),
        "slots": n_slots,
        "complete": len(bracket) == n_slots,
    }


def staffing_shortfall(
    job_demands: Sequence[float],
    worker_capacity: Sequence[float],
    allowed: Iterable[Tuple[int, int]],
) -> List[Tuple[int, float]]:
    """Jobs that cannot be fully staffed, with the hours left over.

    The unmet hours are the min cut way of seeing who is short: a job is short
    exactly when its source edge carries less than its demand. Reaching the sink
    would need more worker capacity than exists in the jobs it can use,
    regardless of how the rest is arranged.
    """
    result = solve_assignment(job_demands, worker_capacity, allowed)
    short = []
    for j, demand in enumerate(job_demands):
        filled = result["per_job"].get(j, 0.0)
        if filled < demand - 1e-9:
            short.append((j, demand - filled))
    return short


def format_schedule(result: Dict[str, object], job_names=None, worker_names=None) -> str:
    """Readable lines for the demo and the CLI."""
    lines = []
    for j, w, hours in result["schedule"]:
        job = job_names[j] if job_names else f"job {j}"
        worker = worker_names[w] if worker_names else f"worker {w}"
        lines.append(f"  {job} -> {worker}: {hours:g} hours")
    return "\n".join(lines)
