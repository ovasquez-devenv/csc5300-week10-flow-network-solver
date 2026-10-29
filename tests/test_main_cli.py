"""Tests for the command line entry point and the visualization outputs."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import main
from core.edmonds_karp import edmonds_karp
from utils.generators import random_sparse


def test_ek_run_prints_and_exits_zero(capsys):
    code = main.main(["--ek", "--generate", "sparse", "--nodes", "200", "--seed", "5300"])
    assert code == 0
    out = capsys.readouterr().out
    assert "flow value" in out
    assert "ok: ok" in out


def test_pr_run(capsys):
    code = main.main(["--pr", "--generate", "sparse", "--nodes", "200", "--seed", "5301"])
    assert code == 0
    assert "Push-relabel" in capsys.readouterr().out


def test_mcf_run(capsys):
    code = main.main(["--mcf", "--generate", "sparse", "--nodes", "120", "--seed", "5302"])
    assert code == 0
    out = capsys.readouterr().out
    assert "total cost" in out
    assert "min_cost: ok" in out


def test_match_run(capsys):
    code = main.main(["--match", "--left", "8", "--right", "8", "--degree", "2", "--seed", "5300"])
    assert code == 0
    assert "matching size" in capsys.readouterr().out


def test_schedule_run(capsys):
    code = main.main(["--schedule", "--jobs", "4", "--workers", "3", "--seed", "5300"])
    assert code == 0
    assert "feasible" in capsys.readouterr().out


def test_segment_run_with_saved_image(tmp_path, capsys):
    from applications.image_seg import make_test_image

    image_path = tmp_path / "img.npy"
    np.save(image_path, make_test_image(16, 5300))
    out_path = tmp_path / "mask.npy"
    code = main.main(["--segment", "--image", str(image_path), "--out", str(out_path)])
    assert code == 0
    assert out_path.exists()
    assert "accuracy" in capsys.readouterr().out


def test_graph_round_trip(tmp_path):
    net = random_sparse(60, 5300)
    path = tmp_path / "graph.json"
    main.save_graph(net, str(path))
    loaded = main.load_graph(str(path))
    assert loaded.n == net.n
    assert loaded.source == net.source
    assert loaded.sink == net.sink
    assert loaded.original_edge_count == net.original_edge_count
    assert edmonds_karp(loaded).value == pytest.approx(edmonds_karp(net.copy()).value)


def test_graph_without_costs_loads(tmp_path):
    data = {"n": 3, "source": 0, "sink": 2, "edges": [[0, 1, 4], [1, 2, 3]]}
    path = tmp_path / "simple.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    net = main.load_graph(str(path))
    assert net.original_edge_count == 2
    assert edmonds_karp(net).value == pytest.approx(3.0)


def test_visualization_file_is_written(tmp_path):
    from visualization.plot_network import draw_cut, draw_network

    net = random_sparse(30, 5300)
    result = edmonds_karp(net)
    png = tmp_path / "net.png"
    out = draw_network(net, str(png), result.flow, title="test")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000

    png2 = tmp_path / "cut.png"
    out2 = draw_cut(net, result.flow, str(png2))
    assert os.path.exists(out2)


def test_matching_plot_is_written(tmp_path):
    from applications.matching import bipartite_matching
    from visualization.plot_network import draw_matching

    edges = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    pairs = bipartite_matching(3, 3, edges)
    out = draw_matching(3, 3, edges, pairs, str(tmp_path / "match.png"))
    assert os.path.exists(out)


def test_segmentation_plot_is_written(tmp_path):
    from applications.image_seg import make_test_image, segment_image
    from visualization.plot_network import draw_segmentation

    image = make_test_image(20, 5300)
    result = segment_image(image, solver="pr")
    out = draw_segmentation(image, result["mask"], str(tmp_path / "seg.png"))
    assert os.path.exists(out)


def test_animation_files_are_written(tmp_path):
    from visualization.animate_flow import animate_edmonds_karp, animate_push_relabel

    net = random_sparse(22, 5300, fan_out=2)
    ek = edmonds_karp(net.copy(), keep_paths=True)
    gif = animate_edmonds_karp(net, ek.stats["paths"], str(tmp_path / "ek.gif"), max_frames=5)
    assert os.path.exists(gif)
    assert os.path.getsize(gif) > 2000

    from core.push_relabel import push_relabel

    pr = push_relabel(net.copy(), trace=True)
    gif2 = animate_push_relabel(net, pr.stats["trace"], str(tmp_path / "pr.gif"), max_frames=5)
    assert os.path.exists(gif2)


def test_no_flags_returns_help(capsys):
    assert main.main([]) == 1
