"""Push-relabel tests: correctness, agreement with Edmonds-Karp, and scaling."""

from __future__ import annotations

import time

import pytest

from core.edmonds_karp import edmonds_karp
from core.network import FlowNetwork
from core.push_relabel import push_relabel
from utils.generators import random_digraph, random_sparse
from utils.validators import validate_flow


@pytest.mark.parametrize("variant", ["highest_label", "fifo"])
def test_known_answer(tiny_network, variant):
    result = push_relabel(tiny_network, variant=variant)
    assert result.value == pytest.approx(23.0)


@pytest.mark.parametrize("variant", ["highest_label", "fifo"])
def test_returns_valid_flow(tiny_network, variant):
    net = tiny_network
    result = push_relabel(net, variant=variant)
    report = validate_flow(net, result.flow, 0, 5, result.value)
    assert report["ok"], report["problems"]


def test_agrees_with_edmonds_karp(seeds):
    """Both solvers must reach the same max flow on every seeded graph."""
    for seed in seeds:
        base = random_digraph(30, 0.22, seed)
        ek = edmonds_karp(base.copy())
        pr = push_relabel(base.copy())
        assert ek.value == pytest.approx(pr.value, abs=1e-6), f"seed {seed}"


def test_both_variants_agree(seeds):
    for seed in seeds:
        base = random_digraph(24, 0.3, seed)
        high = push_relabel(base.copy(), variant="highest_label")
        fifo = push_relabel(base.copy(), variant="fifo")
        assert high.value == pytest.approx(fifo.value, abs=1e-6), f"seed {seed}"


def test_flow_conserves_everywhere_except_source_and_sink(seeds):
    for seed in seeds:
        net = random_digraph(20, 0.3, seed)
        result = push_relabel(net)
        error = net.conservation_error(result.flow, 0, net.n - 1)
        assert error == pytest.approx(0.0, abs=1e-9), f"seed {seed}"


def test_profile_is_recorded_and_monotone():
    """The convergence profile must exist and the sink excess never drops.

    Flow at the sink only grows during push-relabel, because nothing ever
    leaves the sink. A profile that goes down would mean a bug in the counters.
    """
    net = random_sparse(400, 5300, fan_out=2)
    result = push_relabel(net)
    profile = result.stats["profile"]
    assert len(profile) > 1
    flows = [f for _, f in profile]
    assert all(b >= a - 1e-9 for a, b in zip(flows, flows[1:]))
    assert result.value == pytest.approx(flows[-1])


def test_trace_records_events():
    net = random_sparse(40, 5300, fan_out=2)
    result = push_relabel(net, trace=True)
    trace = result.stats["trace"]
    assert trace
    kinds = {event[0] for event in trace}
    assert "push" in kinds
    # Every push entry is (push, u, v, amount, excess_v).
    for event in trace[:20]:
        assert event[0] in ("push", "relabel", "gap")


def test_scales_to_ten_thousand_nodes():
    """The real requirement: 10,000 nodes has to finish quickly."""
    net = random_sparse(10000, 5300)
    start = time.perf_counter()
    result = push_relabel(net)
    elapsed = time.perf_counter() - start
    assert result.value > 0
    assert elapsed < 60.0, f"took {elapsed:.1f} seconds"


def test_handles_source_with_no_edges():
    net = FlowNetwork(4, 0, 3)
    net.add_edge(1, 3, 5.0)
    result = push_relabel(net)
    assert result.value == 0.0


def test_handles_sink_with_no_incoming_edges():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0)
    assert push_relabel(net).value == 0.0


def test_unknown_variant_raises(tiny_network):
    with pytest.raises(ValueError):
        push_relabel(tiny_network, variant="relabel_to_front_typo")
