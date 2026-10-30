# CSC 5300 Week 10: Network Flows, Matching, and Large-Scale Graph Optimization

Omar Vasquez
Concordia University Texas MCS

This project is a modular flow network solver. It implements Edmonds-Karp,
push-relabel, and min cost max flow on one shared network structure, then uses
those solvers for three applications: bipartite matching, image segmentation by
graph cuts, and scheduling. It also draws the graphs and animates the algorithms
to files, and it benchmarks all three solvers on graphs up to 50,000 nodes.

The technical report is in `analysis/ch9_report.md`. The file name is what the
assignment asks for; the report itself says it is the Week 10 report.

## What is in here

```
FlowNetworkSolver/
├── core/
│   ├── network.py          residual graph, forward and reverse edges, cut helpers
│   ├── edmonds_karp.py     BFS shortest augmenting path
│   ├── push_relabel.py     highest label and FIFO variants, with gap and global relabel
│   └── min_cost_flow.py    successive shortest paths with Johnson potentials
├── applications/
│   ├── matching.py         bipartite matching as max flow, plus weighted matching
│   ├── image_seg.py        graph cut segmentation of a grayscale image
│   └── scheduling.py       job assignment, time slots, tournament brackets
├── visualization/
│   ├── plot_network.py     static drawings with capacity and flow labels, and the cut
│   └── animate_flow.py     GIF animations of both solvers, saved to a file
├── utils/
│   ├── generators.py       seeded sparse, dense, bipartite, and grid graphs
│   └── validators.py       capacity, conservation, residual, and cost checks
├── benchmarks/
│   └── week10_flow_benchmark.py
├── examples/
│   ├── week10_demo.py      a short guided run of every part
│   ├── sample_network.json  a 38 node graph the CLI can load
│   └── sample_image.npy    a synthetic grayscale image
├── tests/                  120 pytest tests
├── results/                the measured CSVs, PNGs, and the run log
├── analysis/
│   └── ch9_report.md       the Week 10 technical report
├── main.py                 the command line interface
├── requirements.txt
└── .gitignore
```

## Setup

Install the packages once:

```
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe -m pip install -r requirements.txt
```

That installs numpy, matplotlib, networkx, scipy, and pytest. networkx is
required, not optional: `visualization/plot_network.py` imports it at the top of
the module and the CLI imports that module, so the tests fail without it.

Every module imports its neighbors by their folder name (`core.network`,
`utils.generators`, and so on), so the project root has to be on the module
search path. In PowerShell, from the project folder:

```
$env:PYTHONPATH="."
```

Set that in the same window before every command below, and set it again if you
close the window. If you use Command Prompt instead, it is `set PYTHONPATH=.`.

## How to run

All commands below are PowerShell, run from the project folder.

Run the tests. There are 120 of them and they take about six seconds:

```
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe -m pytest -v
```

Run the demo, which prints a small worked example of every algorithm and
application:

```
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe examples\week10_demo.py
```

Run the benchmark. The full run took 2,887 seconds, about 48 minutes, on my
Windows PC, because it runs three solvers over two graph families
at five sizes with five seeds each plus the repeated Edmonds-Karp timeouts. A
quick pass at small sizes takes about two minutes:

```
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe benchmarks\week10_flow_benchmark.py
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe benchmarks\week10_flow_benchmark.py --quick
```

Use the command line interface:

```
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --ek --graph examples\sample_network.json --viz out\ek.png
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --pr --generate sparse --nodes 5000 --seed 5300
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --mcf --generate dense --nodes 2000 --seed 5301
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --match --left 12 --right 12 --degree 3 --seed 5300
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --segment --image examples\sample_image.npy --out out\mask.npy --viz out\seg.png
C:\Users\elsan\AppData\Local\Programs\Python\Python314\python.exe main.py --schedule --jobs 5 --workers 4 --seed 5300
```

The flags are `--ek`, `--pr`, and `--mcf` for the three solvers, and `--match`,
`--segment`, and `--schedule` for the three applications. `--viz` writes a PNG,
`--animation` writes a GIF, and `--out` writes a mask or a graph file. Every
solver run prints its answer and then runs the validators on it.

Graph files are JSON:

```
{"n": 3, "source": 0, "sink": 2, "edges": [[0, 1, 4], [1, 2, 3]]}
```

A fourth number on an edge is its cost. Names are optional.

## Seeds

`benchmarks/week10_flow_benchmark.py` uses seeds 5300 through 5304 for every
cell and reports the mean and the standard deviation across those five runs.
Every generator builds its own numpy generator or `random.Random` from the seed
it is given, so the graphs come out identical on every machine. Runtimes vary a
little between machines; the flow values, the augmentation counts, and the
costs do not.

## What each part does

`core/network.py` keeps the residual capacity of every edge in a flat list and
stores each edge next to its reverse, so the reverse is found with one xor. It
provides `add_edge`, which creates the reverse edge automatically, plus
`get_residual_graph`, `flow_map`, `min_cut`, and the conservation and capacity
helpers the validators use.

`core/edmonds_karp.py` finds the shortest augmenting path by breadth first
search and pushes the bottleneck along it. It records every bottleneck, every
path length, and every path, which is what the animation draws.

`core/push_relabel.py` implements the highest label variant and a FIFO variant.
It uses global relabeling and the gap heuristic, both of which matter a lot at
50,000 nodes. It runs a final pass that returns leftover excess to the source,
so the answer is a real flow that passes conservation checks, not just a
preflow.

`core/min_cost_flow.py` uses successive shortest paths with potentials, which
is Johnson's trick, so Dijkstra can be used on every round even when residual
edges have negative costs. With all costs non negative the potentials start at
zero, which avoids a Bellman-Ford pass on large graphs.

`applications/matching.py` turns bipartite matching into max flow with unit
capacities. `maximum_weight_matching` does the same with costs and min cost
flow. `applications/image_seg.py` builds the pixel graph with similarity edges
and source and sink likelihood edges, then takes the cut as the mask.
`applications/scheduling.py` covers job to worker assignment with capacities,
time slot placement, and tournament brackets.

`visualization/` writes PNG and GIF files with the Agg backend, so nothing needs
a display. MP4 output through ffmpeg is attempted if the file name ends in
`.mp4`, and it falls back to GIF when ffmpeg is missing.

`utils/validators.py` checks capacity, conservation, flow value, the residual
graph, the min cut, and min cost feasibility. The benchmark refuses to record a
result that fails a check.

## Results

`results/` holds what the benchmark measured, mean and standard deviation over
seeds 5300 to 5304, with a 120 second budget per run:

- `runtime_vs_size.png`, `memory_vs_size.png`, `augmentations_vs_size.png`, `push_relabel_convergence.png`
- `benchmark_summary.csv`, one row per family, size, and solver, with mean and standard deviation across the five seeds
- `benchmark_runs.csv`, one row per individual run
- `convergence_profiles.csv`, the push-relabel samples behind the convergence plot, labelled by family and size
- `benchmark_run.log`, the console output of the run

Runtime in milliseconds, mean followed by the standard deviation in parentheses.
"timed out" means the solver did not finish within the 120 second budget at that
size, so there is no measurement to report. The row is still in the CSV with a
timeout status.

| nodes | sparse EK | sparse PR | sparse MCF | dense EK | dense PR | dense MCF |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 68.5 (7.5) | 18.5 (3.2) | 373.1 (28.2) | 455.7 (25.6) | 10.6 (6.4) | 2,362.1 (240.2) |
| 5,000 | 1,723.7 (207.8) | 133.6 (23.2) | 12,338.5 (2,019.1) | 18,199.5 (2,015.7) | 75.7 (69.3) | 72,475.8 (5,743.2) |
| 10,000 | 12,298.8 (1,317.5) | 372.9 (98.9) | not run | 66,482.5 (1,761.8) | 240.6 (335.1) | not run |
| 25,000 | 81,912.9 (4,725.0) | 1,429.7 (147.7) | not run | timed out | 634.5 (903.8) | not run |
| 50,000 | timed out | 3,993.6 (610.6) | not run | timed out | 3,060.6 (3,487.2) | not run |

EK is Edmonds-Karp, PR is push-relabel, and MCF is min cost flow. Min cost flow
was only run up to 5,000 nodes, which is why the two larger sizes say not run.

Push-relabel peak memory was 4.2 MB at 1,000 sparse nodes, 15.3 MB at 10,000,
33.4 MB at 25,000, and 61.8 MB at 50,000. On the dense family it was 5.6 MB at
1,000 nodes and 144.6 MB at 50,000.

Every solver that finished a graph returned the same flow value. That holds for
all 35 graph and seed combinations where two or more solvers ran, with no
exceptions.

The numbers in `analysis/ch9_report.md` come from those files.

## Notes and limits

Edmonds-Karp does not reach the assignment's target size. It ran out of the 120
second budget on all five seeds at 50,000 sparse nodes, 25,000 dense nodes, and
50,000 dense nodes. Those rows stay in the CSV with a timeout status and no
runtime, so nothing is hidden. At 25,000 sparse nodes it needed 81,912.9 ms
against 1,429.7 ms for push-relabel.

Min cost flow is the slowest per node, which is expected: it solves a harder
problem, and every augmentation is a Dijkstra search. The benchmark stops it at
5,000 nodes, where dense graphs already take 72,475.8 ms.

The dense family uses a fan out of 8 rather than a fully connected layer. A
fully connected layered graph at 50,000 nodes would have about 11 million edges,
which no implementation of these algorithms finishes quickly, and it is not a
shape that shows up in practice.

The runtimes are from my Windows PC with Python 3.14. They will differ on
other machines. The flow values and the augmentation counts will not, because the
generators are seeded.

Graphs are built to be acyclic with a single source and a single sink, matching
the assignment's description of network flow instances.
