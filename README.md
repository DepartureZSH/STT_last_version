# ITC 2019 University Course Timetabling Solver

A Mixed-Integer Programming (MIP) solver for the [International Timetabling Competition 2019](https://www.itc2019.org/) (ITC 2019) problem, extended with a two-layer hybrid solver that uses random sampling (and planned MARL) to pre-solve computationally expensive "special constraints" before handing the reduced problem to Gurobi.

---

## Problem Overview

The ITC 2019 problem requires simultaneously assigning every course section (class) a **time slot** and **room**, subject to:

- **Hard constraints** — must be satisfied: room capacity, no-overlap, same/different time, same/different room, precedence, travel time, etc.
- **Soft constraints** — minimise penalty: preferred times, preferred rooms, distributions.
- **Special constraints** — structurally hard to model in MIP: `MaxDays`, `MaxDayLoad`, `MaxBreaks`, `MaxBlock`. These generate auxiliary integer variables per (week, day) pair and are the primary driver of model size on large instances.

---

## Architecture

### Motivation

The four special constraints create auxiliary variables that cause the MIP model to exceed available memory on large instances (e.g., `pu-proj-fal19` requires > 256 GB RAM for the full model). The hybrid solver eliminates this bottleneck by pre-solving special constraints outside the MIP.

### Two-Layer Hybrid Design

```
Instance XML
    │
    ├── Phase 1: SpecialConstraintDecomposer
    │       Union-Find groups classes sharing special constraints
    │       into independent "communities"
    │       Classifies every class as: fixed | community | free
    │
    ├── Layer 1: CommunitySolver  (random or MARL)
    │       Samples (time, room) assignments for each community
    │       Checks special constraints internally before passing to MIP
    │       Tracks no-good cuts from MIP feedback
    │
    └── Layer 2: MIPSolver (Gurobi)
            Receives fixed assignments for fixed + community classes
            Solves remaining free variables
            On INFEASIBLE: computes IIS, returns implicated classes
            On FEASIBLE:   outputs solution XML

Feedback loop:
    MIP INFEASIBLE → get_infeasibility_info() → add_no_good() → resample → retry
```

### Class Classification

| Class type | Condition | Handling |
|---|---|---|
| `fixed` | Exactly 1 valid (time, room) option | Pinned directly before MIP |
| `community` | Member of ≥1 special-constraint community | Sampled by Layer 1 |
| `free` | All other classes | Solved as MIP variables |

---

## Repository Layout

```
.
├── README.md
├── CLAUDE.md                   # Detailed developer reference
├── config.yaml                 # Solver configuration
├── main.py                     # CLI entry point
│
├── data/
│   ├── source/
│   │   ├── instances/          # ITC 2019 benchmark instances (.xml)
│   │   ├── test/               # Small test instances
│   │   ├── early/ middle/ late/
│   └── solutions/              # Reference solutions
│
├── src/
│   ├── MIP/
│   │   └── solver.py           # Gurobi MIP solver (primary solver)
│   │
│   ├── hybrid/
│   │   ├── special_constraint_decomposer.py  # Phase 1: community detection
│   │   ├── community_solver_base.py          # Abstract interface (ABC)
│   │   ├── random_community_solver.py        # Phase 2: random sampler
│   │   ├── marl_community_solver.py          # Phase 5: MARL stub (interfaces ready)
│   │   └── hybrid_solver.py                 # Phase 4: orchestration loop
│   │
│   └── utils/
│       ├── dataReader.py       # PSTTReader: parses ITC 2019 XML
│       ├── constraints.py
│       ├── solutionReader.py
│       └── solutionWriter.py
│
└── tests/
    ├── phase1_decomposer_test.ipynb       # Community detection (no Gurobi needed)
    ├── phase2_random_solver_test.ipynb    # Random sampler (no Gurobi needed)
    ├── phase2_random_solver_test.py
    ├── phase3_mip_extensions_test.ipynb   # MIP integration (Gurobi required)
    ├── phase3_mip_extensions_test.py
    ├── phase5_marl_solver_test.ipynb      # MARL interface (no Gurobi needed)
    ├── phase5_marl_solver_test.py
    └── run_phase1_tests.py                # Batch test across all 96 instances
```

---

## Implementation Phases

| Phase | File | Status | Description |
|---|---|---|---|
| 1 | `special_constraint_decomposer.py` | Done | Union-Find community detection, class classification |
| 2 | `random_community_solver.py` | Done | Constraint-ordered random sampler, no-good tracking |
| 3 | `solver.py` (extensions) | Done | `fix_assignments`, `get_infeasibility_info`, `reset_fixed` |
| 4 | `hybrid_solver.py` | Done | Orchestration loop: sample → fix → MIP → feedback |
| 5 | `marl_community_solver.py` | Interface ready | MARL agents (train/sample/update_reward to implement) |
| 6 | `main.py` | Done | CLI entry point for MIP and hybrid modes |

---

## Quick Start

### Prerequisites

```bash
pip install gurobipy torch networkx tqdm pyyaml pandas
```

A valid **Gurobi licence** is required for Phase 3, 4, and 6 (any mode that runs the MIP solver).

### Pure MIP Mode

```bash
python main.py --instance data/source/test/bet-sum18.xml --technique MIP
```

### Hybrid Mode (random pre-solver + MIP)

```bash
python main.py --instance data/source/instances/agh-fal17.xml --technique hybrid
```

### Common Options

```bash
# Override time limit per MIP call (seconds)
python main.py --instance data/source/instances/agh-fal17.xml \
    --technique hybrid --time-limit 300

# Custom output path
python main.py --instance data/source/test/bet-sum18.xml \
    --technique MIP --output solutions/bet-sum18.xml

# Custom config file
python main.py --instance ... --config my_config.yaml
```

---

## Configuration (`config.yaml`)

```yaml
train:
  MIP:
    time_limit:    3600   # seconds per MIP call
    Threads:       8
    MIPGap:        0.01   # 1% optimality gap
    PoolSolutions: 10

hybrid:
  max_iterations:             100    # outer loop cap
  max_attempts_per_community: 1000   # random sampler attempts per community
  rebuild_mip_every:          1      # rebuild MIP model every N iterations
                                     # increase to amortise build cost on large instances
  seed:                       42

config:
  technique:   "MIP"          # "MIP" or "hybrid"
  author:      "anonymous"
  institution: "anonymous"
  country:     "anonymous"
  include_students: false
```

---

## Running Tests

Tests that do **not** require a Gurobi licence (run anywhere):

```bash
# Phase 1: community detection — batch tests all 96 instances
python tests/run_phase1_tests.py

# Phase 2: random sampler
python tests/phase2_random_solver_test.py

# Phase 5: MARL interface conformance
python tests/phase5_marl_solver_test.py
```

Tests that **require** a Gurobi licence (run on licensed machine):

```bash
# Phase 3: fix_assignments / get_infeasibility_info / reset_fixed
python tests/phase3_mip_extensions_test.py
```

Or open the corresponding `.ipynb` notebooks in Jupyter for interactive output.

---

## Key Data Structures

### Time option

Each class time option is a 4-tuple `(weeks_bits, days_bits, start, length)`:

```
weeks_bits : str  — binary, '1' = week active  e.g. "111111" = all 6 weeks
days_bits  : str  — binary, '1' = day active   e.g. "0100000" = Tuesday only
start      : int  — slot index within a day (288 slots/day, 5 min each)
                    slot 96 = 8:00 AM
length     : int  — number of consecutive 5-min slots  e.g. 24 = 2 hours
```

### Special constraint types

| Type | Parameter | Meaning |
|---|---|---|
| `MaxDays(N)` | N = max days | Classes must span ≤ N distinct days |
| `MaxDayLoad(S)` | S = max slots/day | Total slot-length per day ≤ S |
| `MaxBreaks(R,S)` | R = max breaks, S = gap threshold | At most R gaps > S slots between blocks |
| `MaxBlock(M,S)` | M = max block length, S = merge gap | No continuous block > M slots |

---

## Special Constraint Communities — Example Results

On instance `agh-fal17.xml` (5081 total classes):

| Category | Count |
|---|---|
| Fixed classes | 341 |
| Community classes | 409 |
| Free classes | 4331 |
| Communities | 31 |

Random sampler result (1000 attempts/community): 28/31 communities satisfied, 3 exhausted, 0 special-constraint violations in returned samples.

---

## MIP Solver Details

Decision variables:

| Variable | Meaning |
|---|---|
| `x[cid, tidx, rid]` | Class `cid` assigned to time `tidx`, room `rid` |
| `y[cid, tidx]` | Class `cid` assigned to time `tidx` (any room) |
| `w[cid, rid]` | Class `cid` assigned to room `rid` (any time) |
| `u[cid]` | Class `cid` is unassigned (slack) |

Objective: multi-priority — (1) minimise unassigned classes; (2) minimise soft-constraint penalty.

Room availability is filtered at variable-creation time using PyTorch tensor intersection (`unavailable_zip`), keeping the model compact.

---

## Extending with MARL (Phase 5)

The MARL interface is fully defined in `community_solver_base.py`. To implement:

1. Subclass `MARLCommunitySolverBase` in `marl_community_solver.py`
2. Implement `train()`, `sample()`, `update_reward()`, `save_model()`, `load_model()`
3. Swap `RandomCommunitySolver` → `MARLCommunitySolver` in `hybrid_solver.py`

The Phase 5 test suite (`phase5_marl_solver_test.py`) has A-group tests (interface conformance, already passing) and B-group tests (training and sampling quality) that activate automatically once `marl.is_trained` is `True`.

Reward signal design guideline:
- `+1.0` per special constraint satisfied
- `-1.0` per special constraint violated
- `-2.0` additional penalty per class implicated in the MIP IIS

---

## Dependencies

| Package | Purpose |
|---|---|
| `gurobipy` | MIP solver (requires licence) |
| `torch` | Tensor operations for room unavailability checks |
| `networkx` | Graph algorithms |
| `tqdm` | Progress bars |
| `pyyaml` | Config file parsing |
| `pandas` | Test result formatting |
