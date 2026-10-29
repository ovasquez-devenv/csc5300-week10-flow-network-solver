"""Image segmentation tests: the cut finds the shape and the mask is consistent."""

from __future__ import annotations

import numpy as np
import pytest

from applications.image_seg import make_test_image, mask_accuracy, segment_image


def test_recovers_the_rectangle():
    """The synthetic image has a bright square, so the mask should find it."""
    image = make_test_image(40, seed=5300)
    truth = image > 0.45
    result = segment_image(image, solver="pr")
    assert mask_accuracy(result["mask"], truth) > 0.95


def test_all_solvers_agree_on_the_cut_value():
    image = make_test_image(24, seed=5301)
    values = {}
    for solver in ("pr", "ek", "mcf"):
        result = segment_image(image, solver=solver)
        values[solver] = result["flow"]
    assert values["pr"] == pytest.approx(values["ek"], rel=1e-6)
    assert values["ek"] == pytest.approx(values["mcf"], rel=1e-6)


def test_mask_shape_matches_image():
    image = make_test_image(17, seed=5302)
    result = segment_image(image, solver="pr")
    assert result["mask"].shape == image.shape
    assert result["mask"].dtype == bool


def test_seeds_land_on_their_side():
    """A pixel forced to be background must end up in the background."""
    image = make_test_image(30, seed=5303)
    fg = [(10, 10), (10, 11), (11, 10)]
    bg = [(0, 0), (0, 1), (1, 0)]
    result = segment_image(image, foreground_seeds=fg, background_seeds=bg, solver="pr")
    for r, c in fg:
        assert result["mask"][r, c], f"foreground seed ({r}, {c}) was cut away"
    for r, c in bg:
        assert not result["mask"][r, c], f"background seed ({r}, {c}) was kept"


def test_cut_capacity_matches_flow():
    image = make_test_image(20, seed=5304)
    result = segment_image(image, solver="pr")
    assert result["cut_capacity"] == pytest.approx(result["flow"], rel=1e-6)


def test_requires_two_dimensional_input():
    with pytest.raises(ValueError):
        segment_image(np.zeros((4, 4, 3)), solver="pr")


def test_unknown_solver_raises():
    image = make_test_image(8, seed=5300)
    with pytest.raises(ValueError):
        segment_image(image, solver="not_a_solver")


def test_noise_seed_changes_the_image():
    a = make_test_image(20, seed=5300)
    b = make_test_image(20, seed=5301)
    assert not np.allclose(a, b)
