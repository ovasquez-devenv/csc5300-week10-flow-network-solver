"""Shared fixtures and path setup for the test suite.

pytest is run from the project root, and every module imports core, utils,
applications, and visualization as top level packages, so the root has to be on
sys.path here as well as in the README instructions.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.network import FlowNetwork  # noqa: E402
from utils.generators import random_digraph, random_sparse  # noqa: E402

SEEDS = [5300, 5301, 5302, 5303, 5304]


@pytest.fixture
def seeds():
    return list(SEEDS)


@pytest.fixture
def tiny_network():
    """The six node network from CLRS with a known max flow of 23."""
    net = FlowNetwork(6, 0, 5)
    net.add_edge(0, 1, 16)
    net.add_edge(0, 2, 13)
    net.add_edge(1, 2, 10)
    net.add_edge(2, 1, 4)
    net.add_edge(1, 3, 12)
    net.add_edge(3, 2, 9)
    net.add_edge(2, 4, 14)
    net.add_edge(4, 3, 7)
    net.add_edge(3, 5, 20)
    net.add_edge(4, 5, 4)
    return net


@pytest.fixture
def random_graphs():
    """A handful of seeded random graphs used across solver tests."""
    return [random_digraph(30, 0.2, seed) for seed in SEEDS]
