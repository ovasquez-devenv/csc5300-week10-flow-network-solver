"""Scheduling tests: feasibility detection, capacities, and slot placement."""

from __future__ import annotations

import pytest

from applications.scheduling import (
    solve_assignment,
    solve_slots,
    staffing_shortfall,
    tournament_bracket,
)


def test_feasible_assignment_meets_every_demand():
    result = solve_assignment(
        job_demands=[4, 3, 5],
        worker_capacity=[3, 4, 5],
        allowed=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)],
    )
    assert result["feasible"]
    assert result["assigned"] == pytest.approx(result["demand"])
    for j, demand in enumerate([4, 3, 5]):
        assert result["per_job"][j] == pytest.approx(demand)


def test_infeasible_assignment_reports_the_shortfall():
    result = solve_assignment([10, 10], [6, 6], [(0, 0), (1, 1)])
    assert not result["feasible"]
    assert result["unmet"] == pytest.approx(8.0)


def test_worker_capacity_is_never_exceeded():
    demands = [5, 5, 5, 5]
    capacity = [4, 4, 4]
    allowed = [(j, w) for j in range(4) for w in range(3)]
    result = solve_assignment(demands, capacity, allowed)
    for w, used in result["per_worker"].items():
        assert used <= capacity[w] + 1e-9


def test_job_demand_is_never_exceeded():
    result = solve_assignment([3, 3], [10, 10], [(0, 0), (0, 1), (1, 0), (1, 1)])
    for j, filled in result["per_job"].items():
        assert filled <= [3, 3][j] + 1e-9


def test_shortfall_lists_the_jobs_that_cannot_be_filled():
    short = staffing_shortfall([4, 3, 5], [2, 2], [(0, 0), (1, 1), (2, 0), (2, 1)])
    jobs = [j for j, _ in short]
    assert jobs == [0, 1, 2]
    assert dict(short)[2] == pytest.approx(5.0)


def test_costed_assignment_still_meets_demands():
    """With costs, the assignment must be cheapest among those that fit."""
    costs = {(0, 0): 1, (0, 1): 9, (1, 0): 9, (1, 1): 1}
    result = solve_assignment([2, 2], [2, 2], [(0, 0), (0, 1), (1, 0), (1, 1)], costs=costs)
    assert result["feasible"]
    assert result["cost"] == pytest.approx(4.0)


def test_slot_placement_uses_each_slot_once():
    result = solve_slots(4, 4, [(0, 0), (1, 1), (2, 2), (3, 3)])
    assert result["all_placed"]
    slots = [k for _, k in result["placed"]]
    assert len(set(slots)) == len(slots)


def test_slot_placement_reports_unplaced_jobs():
    result = solve_slots(3, 2, [(0, 0), (1, 0), (2, 1)])
    assert result["count"] == 2
    assert not result["all_placed"]


def test_tournament_bracket_completes():
    eligible = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)]
    result = tournament_bracket(4, 4, eligible)
    assert result["complete"]
    assert len(set(result["bracket"].values())) == 4


def test_tournament_bracket_can_fail():
    """Both teams can only use slot 0 and every slot holds one team.

    Slot 0 fills, one team is left out, so the bracket cannot be completed even
    though both edges exist. This is the max flow being limited by the cut.
    """
    result = tournament_bracket(2, 3, [(0, 0), (1, 0)])
    assert not result["complete"]
    assert result["filled"] == 1


def test_unknown_pair_raises():
    with pytest.raises(IndexError):
        solve_assignment([1], [1], [(3, 0)])
