"""Checks that a claimed flow really is a flow.

Every function here returns a report instead of raising, so the CLI and the
tests can print or assert on it. The checks follow the definition:

1. Capacity. No edge carries more than its capacity, and no edge carries a
   negative amount.
2. Conservation. At every node other than the source and the sink, the flow in
   equals the flow out.
3. Value. The flow leaving the source equals the flow entering the sink, and
   that common value is what the solver claims.
4. Residual. Every edge in the residual graph corresponds to real leftover
   capacity, and no edge that should have residual capacity is missing.
5. Cut. The value of the flow equals the capacity of the min cut it induces,
   which is the max flow min cut theorem doing the checking.
6. Cost. A min cost flow is feasible and its cost matches the reported cost.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.network import FlowNetwork

TOL = 1e-9


def check_capacity(net: FlowNetwork, flow: Dict[Tuple[int, int], float]) -> Tuple[bool, List[str]]:
    """Every edge stays within its capacity and never goes negative."""
    problems: List[str] = []
    bad = net.capacity_violations(flow)
    for u, v, f, cap in bad:
        problems.append(f"edge ({u}, {v}) carries {f:g} with capacity {cap:g}")
    return not problems, problems


def check_conservation(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: int,
    sink: int,
) -> Tuple[bool, List[str]]:
    """Flow in equals flow out at every node except source and sink."""
    problems: List[str] = []
    balance = net.flow_balance(flow)
    for node, value in balance.items():
        if node in (source, sink):
            continue
        if abs(value) > TOL:
            problems.append(f"node {node} has net outflow {value:g}")
    return not problems, problems


def check_value(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: int,
    sink: int,
    claimed: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """The flow value matches the claimed number and the source and sink agree."""
    problems: List[str] = []
    balance = net.flow_balance(flow)
    out_of_source = balance[source]
    into_sink = -balance[sink]
    if abs(out_of_source - into_sink) > 1e-6:
        problems.append(f"source sends {out_of_source:g} but sink receives {into_sink:g}")
    if claimed is not None and abs(out_of_source - claimed) > 1e-6:
        problems.append(f"claimed value {claimed:g} but measured {out_of_source:g}")
    return not problems, problems


def check_residual(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: int,
    sink: int,
) -> Tuple[bool, List[str]]:
    """Every residual edge has positive capacity and the cut matches the flow.

    The second half is the useful part: it recomputes the residual graph from
    the flow, takes the reachable set from the source, and checks that the cut
    capacity equals the flow value. A flow that misses an augmenting path fails
    here.
    """
    problems: List[str] = []
    balance = net.flow_balance(flow)
    value = balance[source]
    _, _, cut = net.min_cut(source, flow)
    if abs(cut - value) > 1e-6:
        problems.append(f"cut capacity {cut:g} does not match flow value {value:g}")

    # Rebuild a residual copy by hand and compare the reachable sets. The copy
    # has to be reset first, because the solver already spent the capacities on
    # the original network and net.copy carries whatever is left.
    residual = net.copy()
    residual.reset()
    caps = residual.capacities
    heads = residual.heads
    for i in range(0, len(heads), 2):
        u = heads[i ^ 1]
        v = heads[i]
        used = flow.get((u, v), 0.0)
        caps[i] = max(0.0, caps[i] - used)
        caps[i ^ 1] = caps[i ^ 1] + used
    reach_manual = residual.reachable_from(source)
    reach_helper = net.reachable_from(source, flow)
    if reach_manual != reach_helper:
        only_manual = sorted(reach_manual - reach_helper)[:5]
        only_helper = sorted(reach_helper - reach_manual)[:5]
        problems.append(f"residual mismatch, only in rebuilt: {only_manual}, only in helper: {only_helper}")
    return not problems, problems


def check_min_cost(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: int,
    sink: int,
    claimed_cost: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """A min cost flow is a valid flow and its cost matches the claim."""
    problems: List[str] = []
    ok, issues = check_capacity(net, flow)
    problems.extend(issues)
    ok2, issues2 = check_conservation(net, flow, source, sink)
    problems.extend(issues2)

    cost = 0.0
    limits = {}
    for edge in net.iter_edges():
        limits[(edge.u, edge.v)] = limits.get((edge.u, edge.v), 0.0) + 1.0
        cost += flow.get((edge.u, edge.v), 0.0) * edge.cost
    if claimed_cost is not None and abs(cost - claimed_cost) > 1e-6 * max(1.0, abs(claimed_cost)):
        problems.append(f"claimed cost {claimed_cost:g} but measured {cost:g}")
    return not problems, problems


def check_optimality_by_cut(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: int,
    sink: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """The sink must be unreachable from the source in the residual graph.

    That is exactly the stopping condition of every max flow algorithm: if a
    residual path from source to sink existed, pushing more flow along it would
    increase the answer. Checking it independently of the solver is the point,
    since the solver's own claim about being finished is what is in question.

    The cut capacity is reported too, because a flow whose cut does not match
    its value is broken in a different way and the message should say which.
    """
    problems: List[str] = []
    reachable = net.reachable_from(source, flow)
    if sink is not None and sink in reachable:
        problems.append(
            f"sink {sink} is still reachable from source {source}, so the flow is not maximum"
        )

    left, _, cut = net.min_cut(source, flow)
    balance = net.flow_balance(flow)
    value = balance[source]
    if abs(cut - value) > 1e-6:
        problems.append(f"cut capacity {cut:g} does not match the flow value {value:g}")

    # Name one crossing edge so the failure is actionable.
    for edge in net.iter_edges():
        if edge.u in left and edge.v not in left:
            remaining = edge.capacity - flow.get((edge.u, edge.v), 0.0)
            if remaining > 1e-9:
                problems.append(
                    f"residual edge ({edge.u}, {edge.v}) crosses the cut with {remaining:g} left"
                )
                break
    return not problems, problems


def _reachable_to_sink(net: FlowNetwork, flow: Dict[Tuple[int, int], float], sink: int) -> set:
    """Nodes that can still send flow to the sink, walking residual edges backward."""
    limits: Dict[Tuple[int, int], float] = {}
    for edge in net.iter_edges():
        key = (edge.u, edge.v)
        limits[key] = limits.get(key, 0.0) + edge.capacity
    reverse: Dict[int, List[Tuple[int, float]]] = {u: [] for u in range(net.n)}
    for (u, v), cap in limits.items():
        reverse[v].append((u, cap - flow.get((u, v), 0.0)))
    seen = {sink}
    stack = [sink]
    while stack:
        v = stack.pop()
        for u, remaining in reverse[v]:
            if remaining > 1e-9 and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def validate_flow(
    net: FlowNetwork,
    flow: Dict[Tuple[int, int], float],
    source: Optional[int] = None,
    sink: Optional[int] = None,
    claimed_value: Optional[float] = None,
    claimed_cost: Optional[float] = None,
    check_maximal: bool = True,
) -> Dict[str, object]:
    """Run every check and return one report.

    The report has a top level ok flag, one entry per check, a flat list of
    problems, and the measured flow value, so a test can assert on any of them.
    """
    s = net.source if source is None else source
    t = net.sink if sink is None else sink
    if s is None or t is None:
        raise ValueError("source and sink must be set")

    results = {}
    problems: List[str] = []

    ok, issues = check_capacity(net, flow)
    results["capacity"] = ok
    problems.extend(issues)

    ok, issues = check_conservation(net, flow, s, t)
    results["conservation"] = ok
    problems.extend(issues)

    ok, issues = check_value(net, flow, s, t, claimed_value)
    results["value"] = ok
    problems.extend(issues)

    ok, issues = check_residual(net, flow, s, t)
    results["residual"] = ok
    problems.extend(issues)

    if check_maximal:
        ok, issues = check_optimality_by_cut(net, flow, s, t)
        results["maximal"] = ok
        problems.extend(issues)

    if claimed_cost is not None:
        ok, issues = check_min_cost(net, flow, s, t, claimed_cost)
        results["min_cost"] = ok
        problems.extend(issues)

    balance = net.flow_balance(flow)
    results["ok"] = all(v for k, v in results.items() if k != "ok")
    results["problems"] = problems
    results["value"] = balance[s]
    return results


def broken_flow(net: FlowNetwork, source: int, sink: int):
    """A deliberately invalid flow, for the test that proves the checkers work.

    It sends one unit along the first edge out of the source while ignoring
    capacity, which breaks both capacity and conservation on any real instance.
    """
    for edge in net.iter_edges():
        if edge.u == source:
            return {(edge.u, edge.v): edge.capacity + 1.0}
    return {(source, sink): 1.0}


def summarize_report(report: Dict[str, object]) -> str:
    """One readable line per check."""
    lines = []
    for key, value in report.items():
        if key in ("problems", "value"):
            continue
        mark = "ok" if value else "FAILED"
        lines.append(f"  {key}: {mark}")
    if report.get("problems"):
        for problem in report["problems"][:10]:
            lines.append(f"    {problem}")
    return "\n".join(lines)
