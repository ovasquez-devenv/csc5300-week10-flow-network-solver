"""Tests that the validators pass good flows and catch broken ones."""

from __future__ import annotations

import pytest

from core.edmonds_karp import edmonds_karp
from core.network import FlowNetwork
from utils.validators import (
    broken_flow,
    check_capacity,
    check_conservation,
    check_min_cost,
    check_optimality_by_cut,
    check_residual,
    check_value,
    summarize_report,
    validate_flow,
)


def test_good_flow_passes_every_check(tiny_network):
    net = tiny_network
    result = edmonds_karp(net)
    report = validate_flow(net, result.flow, 0, 5, result.value)
    assert report["ok"]
    assert report["problems"] == []
    assert report["value"] == pytest.approx(23.0)


def test_broken_flow_is_caught(tiny_network):
    """The deliberate capacity overflow must fail, or the checks are useless."""
    net = tiny_network
    bad = broken_flow(net, 0, 5)
    report = validate_flow(net, bad, 0, 5)
    assert not report["ok"]
    assert not report["capacity"]
    assert report["problems"]


def test_conservation_violation_is_caught():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0)
    net.add_edge(1, 2, 5.0)
    ok, problems = check_conservation(net, {(0, 1): 4.0, (1, 2): 2.0}, 0, 2)
    assert not ok
    assert any("node 1" in p for p in problems)


def test_negative_flow_is_caught():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 5.0)
    net.add_edge(1, 2, 5.0)
    ok, problems = check_capacity(net, {(0, 1): -1.0})
    assert not ok
    assert problems


def test_value_mismatch_is_caught(tiny_network):
    net = tiny_network
    result = edmonds_karp(net)
    ok, problems = check_value(net, result.flow, 0, 5, claimed=result.value + 5.0)
    assert not ok
    assert problems


def test_non_maximal_flow_is_caught(tiny_network):
    """Half the flow through one path leaves residual edges across the cut."""
    net = tiny_network
    partial = {(0, 1): 10.0, (1, 3): 10.0, (3, 5): 10.0}
    ok, problems = check_optimality_by_cut(net, partial, 0, 5)
    assert not ok
    assert problems


def test_residual_check_passes_on_real_flow(tiny_network):
    net = tiny_network
    result = edmonds_karp(net)
    ok, problems = check_residual(net, result.flow, 0, 5)
    assert ok, problems


def test_min_cost_check_passes_and_fails():
    from core.min_cost_flow import min_cost_flow

    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 4.0, 2.0)
    net.add_edge(1, 2, 4.0, 3.0)
    result = min_cost_flow(net)
    ok, problems = check_min_cost(net, result.flow, 0, 2, result.stats["cost"])
    assert ok, problems
    ok2, problems2 = check_min_cost(net, result.flow, 0, 2, result.stats["cost"] + 100.0)
    assert not ok2
    assert problems2


def test_summary_lists_each_check():
    net = FlowNetwork(3, 0, 2)
    net.add_edge(0, 1, 4.0)
    net.add_edge(1, 2, 4.0)
    result = edmonds_karp(net)
    text = summarize_report(validate_flow(net, result.flow, 0, 2, result.value))
    assert "capacity" in text
    assert "conservation" in text
    assert "maximal" in text


def test_validator_requires_source_and_sink():
    net = FlowNetwork(3)
    net.add_edge(0, 1, 1.0)
    with pytest.raises(ValueError):
        validate_flow(net, {}, None, None)
