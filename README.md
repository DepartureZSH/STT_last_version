# ITC 2019 University Course Timetabling Solver

A Mixed-Integer Programming (MIP) solver for the [International Timetabling Competition 2019](https://www.itc2019.org/) (ITC 2019) problem, with two extension modes: a **hybrid** solver that uses random sampling to pre-solve structurally expensive constraints, and a **divide-and-conquer** solver that partitions large instances into tractable sub-problems.

---

## Problem Overview

The ITC 2019 problem requires simultaneously assigning every course section (class) a **time slot** and **room**, subject to:

- **Hard constraints** — must be satisfied: room capacity, no-overlap, same/different time, same/different room, precedence, travel time, etc.
- **Soft constraints** — minimise penalty: preferred times, preferred rooms, distributions.
- **Special constraints** — structurally hard to model in MIP: `MaxDays`, `MaxDayLoad`, `MaxBreaks`, `MaxBlock`. These generate auxiliary integer variables per (week, day) pair and are the primary driver of model size on large instances.

---

## Solver Techniques

### 1. Pure MIP (`--technique MIP`)

Builds a single Gurobi model for the entire instance and solves it directly. Works well for small and medium instances. On large instances (e.g., `lums-spr18`) the model may have > 500 K binary variables and can exhaust available memory.

### 2. Hybrid (`--technique hybrid`)

A two-layer approach for instances whose size is driven by **special constraints**:

```
Instance XML
    │
    ├── Phase 1: SpecialConstraintDecomposer
    │       Union-Find groups classes sharing special constraints
    │       into independent "communities"
    │       Classifies every class as: fixed | community | free
    │
    ├── Layer 1: CommunitySolver  (random sampler / MARL)
    │       Samples (time, room) assignments for each community
    │       Checks special constraints internally
    │       Tracks no-good cuts from MIP feedback
    │
    └── Layer 2: MIPSolver (Gurobi)
            Receives fixed assignments for fixed + community classes
            Solves remaining free variables
            On INFEASIBLE: returns IIS-implicated classes → feedback loop
            On FEASIBLE:   outputs solution XML

Feedback: MIP INFEASIBLE → get_infeasibility_info() → add_no_good() → resample
```

### 3. Divide & Conquer (`--technique divide_conquer`)

Designed for very large instances where even the special-constraint decomposition leaves sub-problems too large for memory. Partitions by the **hard-constraint graph** instead, guaranteeing conflict-free room assignment by construction.

---

## Divide & Conquer Design Flow

```
Instance XML
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 1 — BFS Partitioning                  │
│  HardConstraintPartitioner                  │
│                                             │
│  Build adjacency graph: classes sharing a   │
│  hard constraint are adjacent.              │
│                                             │
│  BFS from each unvisited class:             │
│  • Add neighbours to queue when discovered  │
│  • Stop when group reaches max_size (50)    │
│  • Overflow queue items: un-mark visited    │
│    so they seed new partitions (orphan fix) │
│                                             │
│  Result: P partitions, each ≤ max_size      │
│  Cross-partition constraints identified     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Step 2 — Sequential Solving                │
│  (largest partition first)                  │
│                                             │
│  For each partition i:                      │
│    a) Build sub-MIP (variables for          │
│       partition classes only)               │
│    b) Room reservation: block all           │
│       (room, time_bits) pairs already       │
│       claimed by partitions 1..i-1          │
│       via x[cid,tidx,rid] = 0              │
│    c) Time reservation: for each cross-     │
│       partition hard constraint (NotOverlap │
│       / SameAttendees / SameTime) where     │
│       another class is already assigned,    │
│       block conflicting y[cid,tidx] = 0    │
│    d) Solve with Gurobi                     │
│    e) Register new (room, time_bits) pairs  │
│       into reserved set                     │
│                                             │
│  Room conflicts: impossible by construction │
│  Cross-partition time conflicts: minimised  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Step 3 — Merge-Repartition Repair Loop     │
│  (up to max_retries rounds)                 │
│                                             │
│  Each round:                                │
│  1. Compute score = room_conflicts          │
│                   + time_violations         │
│                   + unassigned_count        │
│  2. If score == 0 → done                   │
│  3. Collect the partition IDs of every      │
│     class involved in any problem           │
│  4. Merge their class sets into one         │
│     super-group; remove their assignments   │
│     from the global solution                │
│  5. Re-partition the super-group with BFS   │
│     using a SHUFFLED traversal order        │
│     (seed = round number) — guarantees a    │
│     structurally different split every      │
│     round, breaking cyclic dead-ends        │
│  6. Re-solve each new sub-partition         │
│     sequentially with room + time           │
│     reservations from the CURRENT global    │
│     assignment state (no stale data)        │
│  7. Commit; update partition registry and   │
│     part_of map with new partition IDs      │
│                                             │
│  Stall guard: stop after 5 consecutive      │
│  rounds with score ≥ previous score         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
            Solution XML
   (saved even if partially unassigned,
    exit code 1 if violations remain)
```

### Key guarantees

| Property | Guarantee |
|---|---|
| Room conflicts within each solve | **Zero** — blocked by sequential room reservation in Steps 2 & 3 |
| Cross-partition time violations | Minimised in Step 2; eliminated by Step 3 merge-repartition |
| Unassigned classes | Resolved by Step 3 (merging internalises conflicting constraints) |
| Memory | Sub-MIP has ≤ `max_partition_size` × rooms × time_options variables |
| Non-repetition | Different seed → different BFS traversal order → different split → no cycling |

### Why merge-repartition instead of per-class backtracking

Per-class backtracking (forbid a time option, re-solve, propagate blockers) has two failure modes:

1. **Cycling** — the same partition re-solves with the same structure, hitting the same dead-end repeatedly.
2. **Cascading blockers** — fixing one class displaces a blocker that in turn needs another blocker resolved, and so on indefinitely.

Merge-repartition avoids both: the partition *boundaries* themselves change, so constraints between formerly-separate groups become intra-partition and are handled directly by the MIP solver.

---

## Repository Layout

```
.
├── README.md
├── CLAUDE.md                   # Developer reference
├── config.yaml                 # Solver configuration
├── main.py                     # CLI entry point
├── validator.py                # Solution validator (ITC 2019 XML)
│
├── data/
│   ├── source/
│   │   ├── instances/          # ITC 2019 benchmark instances (.xml)
│   │   ├── test/               # Small test instances
│   │   └── early/ middle/ late/
│   └── solutions/              # Reference solutions
│
├── solutions/                  # Output solutions from solver runs
│
└── src/
    ├── MIP/
    │   └── solver.py           # Gurobi MIP solver
    │                           #   build_model()  — full instance
    │                           #   build_submodel() — partition subset
    │                           #   reserve_room_times() — room blocking
    │                           #   add_preassigned_time_constraints()
    │                           #   forbid_time_room() / reset_forbid()
    │                           #   fix_assignments() / reset_fixed()
    │                           #   get_infeasibility_info()
    │
    ├── hybrid/
    │   ├── hard_constraint_partitioner.py  — BFS graph partitioner
    │   ├── divide_conquer_solver.py        — D&C orchestration (Steps 1-3)
    │   ├── hybrid_solver.py               — Hybrid orchestration loop
    │   ├── special_constraint_decomposer.py — Community detection (Union-Find)
    │   ├── community_solver_base.py        — Abstract interface (ABC)
    │   ├── random_community_solver.py      — Random sampler
    │   └── marl_community_solver.py        — MARL stub
    │
    └── utils/
        ├── dataReader.py       # PSTTReader: parses ITC 2019 XML
        ├── constraints.py
        ├── solutionReader.py
        └── solutionWriter.py
```

---

## Quick Start

### Prerequisites

```bash
pip install gurobipy torch networkx tqdm pyyaml pandas
```

A valid **Gurobi licence** is required for any mode that calls the MIP solver.

### Pure MIP

```bash
python main.py --instance data/source/test/bet-sum18.xml --technique MIP
```

### Hybrid (random pre-solver + MIP)

```bash
python main.py --instance data/source/instances/agh-fal17.xml --technique hybrid
```

### Divide & Conquer (large instances)

```bash
python main.py --instance data/source/instances/lums-spr18.xml --technique divide_conquer
```

### Common Options

```bash
# Override time limit per MIP call (seconds)
python main.py --instance ... --technique divide_conquer --time-limit 300

# Custom output path
python main.py --instance ... --technique MIP --output solutions/my_sol.xml

# Custom config file
python main.py --instance ... --config my_config.yaml
```

---

## Configuration (`config.yaml`)

```yaml
train:
  MIP:
    time_limit:    300     # seconds per sub-MIP call (D&C) / full solve (MIP)
    Threads:       8
    MIPGap:        0.01    # 1% optimality gap
    PoolSolutions: 10

hybrid:
  max_iterations:             100
  max_attempts_per_community: 1000
  rebuild_mip_every:          1
  seed:                       42

divide_conquer:
  max_partition_size:  50     # max classes per BFS partition
  max_retries:         100    # merge-repartition round budget
  use_soft_edges:      false  # include soft constraints in BFS graph
  prefer_larger_first: true   # solve larger partitions first

config:
  technique:        "divide_conquer"   # "MIP" | "hybrid" | "divide_conquer"
  author:           "anonymous"
  institution:      "anonymous"
  country:          "anonymous"
  include_students: false
```

---

## Implementation Phases

| Phase | File(s) | Status | Description |
|---|---|---|---|
| 1 | `special_constraint_decomposer.py` | Done | Union-Find community detection, class classification |
| 2 | `random_community_solver.py` | Done | Constraint-ordered random sampler, no-good tracking |
| 3 | `solver.py` (extensions) | Done | `fix_assignments`, `get_infeasibility_info`, `reset_fixed`, `reserve_room_times`, `add_preassigned_time_constraints`, `build_submodel` |
| 4 | `hybrid_solver.py` | Done | Orchestration loop: sample → fix → MIP → feedback |
| 5 | `marl_community_solver.py` | Interface ready | MARL agents (train/sample/update_reward to implement) |
| 6 | `divide_conquer_solver.py` + `hard_constraint_partitioner.py` | Done | BFS partition → sequential solve → merge-repartition repair |

---

## Key Data Structures

### Time option (4-tuple)

```
(weeks_bits, days_bits, start, length)

weeks_bits : str  — binary string, '1' = week active  e.g. "111111" (6 weeks)
days_bits  : str  — binary string, '1' = day active   e.g. "0100000" (Tuesday)
start      : int  — slot index within a day (288 slots/day, 5 min each)
                    slot 96 = 8:00 AM
length     : int  — number of consecutive 5-min slots  e.g. 24 = 2 hours
```

Two time options **conflict** when all three overlap: week bits AND, day bits AND, and slot ranges intersect.

### MIP decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `x[cid, tidx, rid]` | {0,1} | Class assigned to time `tidx` in room `rid` |
| `y[cid, tidx]` | {0,1} | Class assigned to time `tidx` (any room) |
| `w[cid, rid]` | {0,1} | Class assigned to room `rid` (any time) |
| `u[cid]` | {0,1} | Class is unassigned (slack, penalised heavily) |

### Partition (D&C)

```python
@dataclass
class Partition:
    id: int
    class_ids: List[str]         # classes in this BFS group
    hard_constraints: List[dict] # hard constraints with ≥2 in-partition classes
    soft_constraints: List[dict] # soft constraints with ≥2 in-partition classes
```

---

## Divide & Conquer — Scalability Notes

On `lums-spr18.xml` (487 classes, ~36% cross-partition hard constraints):

- Full MIP: out-of-memory (> available RAM)
- D&C with `max_partition_size=50`: ~117 partitions, each sub-MIP tractable
- Sequential solve: ~120 Gurobi calls
- Room conflicts after Step 2: **0** (guaranteed by sequential reservation)
- Remaining violations: eliminated by Step 3 merge-repartition

Cross-partition hard constraints (~36%) are handled by:
1. **Time reservation** (Step 2c): blocks conflicting time options before each sub-MIP
2. **Merge-repartition** (Step 3): when violations remain, merges offending partitions and re-partitions with a fresh random seed so constraints between formerly-separate groups become intra-partition

---

## Special Constraint Types

| Type | Parameter | Meaning |
|---|---|---|
| `MaxDays(N)` | N = max days | Classes must span ≤ N distinct days |
| `MaxDayLoad(S)` | S = max slots/day | Total slot-length per day ≤ S |
| `MaxBreaks(R,S)` | R = max breaks, S = gap threshold | At most R gaps > S slots between consecutive blocks |
| `MaxBlock(M,S)` | M = max block length, S = merge gap | No continuous block > M slots |

---

## Extending with MARL (Phase 5)

The MARL interface is defined in `community_solver_base.py`. To implement:

1. Subclass `MARLCommunitySolverBase` in `marl_community_solver.py`
2. Implement `train()`, `sample()`, `update_reward()`, `save_model()`, `load_model()`
3. Swap `RandomCommunitySolver` → `MARLCommunitySolver` in `hybrid_solver.py`

Reward signal design guideline:
- `+1.0` per special constraint satisfied
- `-1.0` per special constraint violated
- `-2.0` additional penalty per class implicated in the MIP IIS

---

## Dependencies

| Package | Purpose |
|---|---|
| `gurobipy` | MIP solver (requires licence) |
| `torch` | Tensor operations for room unavailability filtering |
| `networkx` | Graph algorithms |
| `tqdm` | Progress bars |
| `pyyaml` | Config file parsing |
| `pandas` | Test result formatting |
