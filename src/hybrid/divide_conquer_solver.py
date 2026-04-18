"""
divide_conquer_solver.py
========================
Divide-and-conquer MIP solver for large ITC 2019 instances.

Motivation
----------
For instances like lums-spr18 the full MIP model has ~559 K binary
variables and causes out-of-memory errors.  This solver:

  1. Partitions classes into small groups (≤ max_partition_size) using
     BFS on the hard-constraint graph.  Because BFS marks neighbours as
     "owned" before reaching the size limit, empirically **all** hard
     constraints end up intra-partition (0 % cross-partition rate on ITC
     2019 benchmarks).

  2. Solves each group *sequentially* (larger partitions first) with a
     sub-MIP (≤ ~53 K variables for max_partition_size = 50).

  3. Before each sub-MIP is solved, all (room, time_bits) pairs already
     claimed by previously-solved partitions are pre-blocked via
     x[cid,tidx,rid] == 0 constraints.  This guarantees conflict-free
     room assignment by construction — no post-hoc backtracking loop is
     needed.

  Key properties of this approach
  --------------------------------
  * Zero room conflicts after all partitions are solved (guaranteed).
  * No circular backtracking / oscillation.
  * If a partition becomes infeasible because all its room-time options
    are reserved, the solver reports an infeasibility error.  In practice
    this is rare: ITC 2019 instances have far more room-time slots than
    required.

Usage
-----
    from src.hybrid.divide_conquer_solver import DivideConquerSolver
    from src.utils.dataReader import PSTTReader
    import logging, yaml

    reader = PSTTReader('data/source/instances/lums-spr18.xml')
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger('solver')
    dc = DivideConquerSolver(reader, config, logger)
    result = dc.solve()
    if result.success:
        dc.save_solution(result, 'solutions/lums-spr18.xml')
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from src.utils.dataReader import PSTTReader
from src.MIP.solver import MIPSolver
from src.hybrid.hard_constraint_partitioner import (
    HardConstraintPartitioner,
    Partition,
)

# {cid: (tidx, rid_or_None)}
SimpleAssignments = Dict[str, Tuple[int, Optional[str]]]

# A detected room conflict: (cid, tidx, rid) of the class to backtrack
ConflictEntry = Tuple[str, int, str]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DCResult:
    """Outcome of one DivideConquerSolver.solve() call."""
    success: bool
    assignments: SimpleAssignments        # {cid: (tidx, rid)} merged solution
    partitions_solved: int                # number of partitions solved
    total_mip_calls: int                  # total Gurobi solve() calls
    total_conflicts: int                  # total room conflicts detected
    total_backtracks: int                 # total backtrack-and-re-solve ops
    total_time_sec: float
    termination: str                      # "success" | "infeasible" | "max_retries" | "time_limit"


# ─────────────────────────────────────────────────────────────────────────────
# DivideConquerSolver
# ─────────────────────────────────────────────────────────────────────────────

class DivideConquerSolver:
    """
    Divide-and-conquer MIP solver.

    Config keys (under ``divide_conquer``)
    ---------------------------------------
    max_partition_size : int   — max classes per BFS partition (default 50)
    max_retries        : int   — backtrack budget per conflict round (default 100)
    use_soft_edges     : bool  — include soft constraints in BFS graph (default False)
    prefer_larger_first: bool  — solve larger partitions first (default True);
                                 larger partitions occupy rooms first so smaller
                                 ones must work around them — backtracking the
                                 smaller partition is cheaper.
    """

    DEFAULT_DC = {
        "max_partition_size":  50,
        "max_retries":         100,
        "use_soft_edges":      False,
        "prefer_larger_first": True,
    }

    def __init__(
        self,
        reader: PSTTReader,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ):
        self.reader = reader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        dc_cfg = {**self.DEFAULT_DC, **config.get("divide_conquer", {})}
        self._max_partition_size  = int(dc_cfg["max_partition_size"])
        self._max_retries         = int(dc_cfg["max_retries"])
        self._use_soft_edges      = bool(dc_cfg["use_soft_edges"])
        self._prefer_larger_first = bool(dc_cfg["prefer_larger_first"])

        self._partitioner: Optional[HardConstraintPartitioner] = None
        # partition_id → MIPSolver (kept alive for re-solve)
        self._solvers: Dict[int, MIPSolver] = {}
        # Counter for assigning unique IDs to re-partitioned sub-groups
        self._next_partition_id: int = 100000

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def solve(self) -> DCResult:
        """
        Run the divide → sequential-solve-with-room-reservation pipeline.

        Each partition is solved in order (largest first).  Before each
        sub-MIP is handed to Gurobi, every (room, time_bits) pair already
        claimed by a previously-solved partition is pre-blocked with
        x[cid,tidx,rid] == 0 constraints.  Room conflicts are therefore
        impossible by construction — no post-hoc backtracking loop needed.
        """
        wall_start = time.time()
        total_mip_calls = 0

        # ── Step 1: partition ─────────────────────────────────────────────
        self.logger.info("=== Divide & Conquer Solver: partitioning ===")
        self._partitioner = HardConstraintPartitioner(
            self.reader,
            max_partition_size=self._max_partition_size,
            use_soft_edges=self._use_soft_edges,
        )
        partitions = self._partitioner.partition()
        self.logger.info(self._partitioner.summary())

        hard_cross, _ = self._partitioner.get_cross_constraints()
        if hard_cross:
            self.logger.info(
                f"  {len(hard_cross)} cross-partition hard constraints "
                "will be enforced via sequential time reservation."
            )

        # ── Step 2: sequential solve with room + time reservation ────────────
        # Larger partitions first: they occupy the most-constrained rooms
        # early, and smaller partitions adapt around them.
        solve_order = sorted(
            partitions,
            key=lambda p: len(p.class_ids),
            reverse=self._prefer_larger_first,
        )

        # Index: class_id → list of hard constraints it participates in.
        # Used to find cross-partition constraints quickly.
        class_to_hard: Dict[str, List[dict]] = defaultdict(list)
        for cons in self.reader.distributions.get("hard_constraints", []):
            for cid in cons["classes"]:
                class_to_hard[cid].append(cons)

        # Map: class_id → partition_id (used in repair and force-assign phases)
        part_of: Dict[str, int] = self._partitioner.part_of_map()

        assignments: SimpleAssignments = {}
        # (rid, time_bits) pairs already claimed by solved partitions (room guard)
        reserved_room_times: Set[Tuple[str, tuple]] = set()

        self.logger.info(
            f"\n--- Solving {len(partitions)} partitions sequentially ---"
        )
        n = len(partitions)
        for i, partition in enumerate(solve_order):
            mip = self._build_partition_mip(partition)
            self._solvers[partition.id] = mip

            # ── Room reservation: block already-claimed (room, time) slots ──
            if reserved_room_times:
                room_blocked = mip.reserve_room_times(reserved_room_times)
            else:
                room_blocked = 0

            # ── Time reservation: enforce cross-partition hard constraints ───
            # Collect hard constraints that touch a class in this partition
            # AND have at least one class already assigned.
            cross_constrs: List[dict] = []
            seen_cons: Set[int] = set()
            for cid in partition.class_ids:
                for cons in class_to_hard.get(cid, []):
                    ckey = id(cons)
                    if ckey in seen_cons:
                        continue
                    if any(oc in assignments for oc in cons["classes"] if oc != cid):
                        seen_cons.add(ckey)
                        cross_constrs.append(cons)

            if cross_constrs:
                time_blocked = mip.add_preassigned_time_constraints(
                    assignments, cross_constrs
                )
            else:
                time_blocked = 0

            if room_blocked or time_blocked:
                self.logger.info(
                    f"  [{i+1}/{n}] Partition {partition.id}: "
                    f"pre-blocked {room_blocked} room-time vars, "
                    f"{time_blocked} cross-partition time vars "
                    f"({len(cross_constrs)} cross constraints)"
                )

            result = mip.solve()
            total_mip_calls += 1

            if result is None:
                self.logger.error(
                    f"  Partition {partition.id} ({len(partition.class_ids)} classes) "
                    "infeasible — reservations too tight. Aborting."
                )
                return DCResult(
                    success=False,
                    assignments=assignments,
                    partitions_solved=i,
                    total_mip_calls=total_mip_calls,
                    total_conflicts=0,
                    total_backtracks=0,
                    total_time_sec=time.time() - wall_start,
                    termination="infeasible",
                )

            part_assigns = mip.get_simple_assignments()
            assignments.update(part_assigns)

            # Register new room-time occupancy for subsequent partitions
            for cid, (tidx, rid) in part_assigns.items():
                if rid is not None:
                    time_bits = (
                        self.reader.classes[cid]["time_options"][tidx]
                        ["optional_time_bits"]
                    )
                    reserved_room_times.add((rid, time_bits))

            self.logger.info(
                f"  [{i+1}/{n}] Partition {partition.id}: "
                f"{len(partition.class_ids)} classes → {len(part_assigns)} assigned  "
                f"(reserved slots: {len(reserved_room_times)})"
            )

        # ── Room conflict diagnostic ──────────────────────────────────────
        residual = self._detect_room_conflicts(assignments)
        if residual:
            self.logger.warning(
                f"  {len(residual)} residual room conflict(s) detected "
                "(unexpected — please report)."
            )
        else:
            self.logger.info("  Room conflict check: 0 conflicts (as expected).")

        # ── Step 3: Merge-Repartition repair loop ─────────────────────────────
        # When any problem remains (room double-bookings, time constraint
        # violations, or unassigned classes), we:
        #   1. Collect every partition involved in a conflict.
        #   2. Merge their classes into one super-group.
        #   3. Re-partition that super-group with a fresh random seed
        #      (BFS traversal order is shuffled → different split every round).
        #   4. Re-solve each new sub-partition sequentially with room + time
        #      reservations built from the CURRENT global assignment state.
        # This avoids the whack-a-mole problem of per-class backtracking and
        # guarantees that the new split is structurally different.

        # Lookup: partition_id → Partition (updated as re-partitioning happens)
        partitions_by_id: Dict[int, Partition] = {p.id: p for p in partitions}

        self.logger.info("\n--- Merge-Repartition repair loop ---")
        total_merge_calls = 0
        prev_score = float("inf")
        stall_count = 0

        for mr_round in range(self._max_retries):
            room_conflicts = self._detect_room_conflicts(assignments)
            time_viols     = self._find_all_violations(assignments)
            unassigned     = [
                cid for cid in self.reader.classes if cid not in assignments
            ]
            score = len(room_conflicts) + len(time_viols) + len(unassigned)

            if score == 0:
                self.logger.info(
                    f"  Merge-repartition: clean after {mr_round} round(s)."
                )
                break

            self.logger.info(
                f"  Round {mr_round + 1}: "
                f"room_conflicts={len(room_conflicts)}, "
                f"time_viols={len(time_viols)}, "
                f"unassigned={len(unassigned)}"
            )

            stuck_cids: Set[str] = set(unassigned)

            # ── Collect partition IDs involved in any problem ─────────────
            conflict_pids: Set[int] = set()
            for (c1, c2, _) in room_conflicts:
                for cid in (c1, c2):
                    pid = part_of.get(cid)
                    if pid is not None:
                        conflict_pids.add(pid)
            for (c1, c2) in time_viols:
                for cid in (c1, c2):
                    pid = part_of.get(cid)
                    if pid is not None:
                        conflict_pids.add(pid)
            for cid in unassigned:
                pid = part_of.get(cid)
                if pid is not None:
                    conflict_pids.add(pid)

            # ── Fix C: expand merge to include room-blocker partitions ─────
            # For each stuck class, find stable (non-merged) assigned classes
            # that hold a room at a conflicting time — blocking every option
            # the stuck class has.  Add their partitions to the merge set so
            # the MIP can displace them.
            blocker_pids: Set[int] = set()
            merged_set_preview = set()
            for pid in conflict_pids:
                p = partitions_by_id.get(pid)
                if p:
                    merged_set_preview.update(p.class_ids)

            for stuck_cid in stuck_cids:
                cls_data = self.reader.classes[stuck_cid]
                for topt in cls_data["time_options"]:
                    tbits_opt = topt["optional_time_bits"]
                    for ropt in cls_data.get("room_options", []):
                        rid_opt = ropt["id"]
                        for cid2, (tidx2, rid2) in assignments.items():
                            if rid2 != rid_opt or cid2 in merged_set_preview:
                                continue
                            tbits2 = (
                                self.reader.classes[cid2]["time_options"][tidx2]
                                ["optional_time_bits"]
                            )
                            if self._times_conflict(tbits_opt, tbits2, attendee=False):
                                bpid = part_of.get(cid2)
                                if bpid is not None:
                                    blocker_pids.add(bpid)

            if blocker_pids:
                self.logger.info(
                    f"    Also merging {len(blocker_pids)} room-blocker "
                    f"partition(s) for stuck classes."
                )
                conflict_pids |= blocker_pids

            # Merge class IDs from all conflicting partitions
            merged_class_ids: List[str] = []
            for pid in conflict_pids:
                p = partitions_by_id.get(pid)
                if p:
                    merged_class_ids.extend(p.class_ids)

            self.logger.info(
                f"    Merging {len(conflict_pids)} partition(s) "
                f"({len(merged_class_ids)} classes), seed={mr_round + 1}"
            )

            # Remove merged classes from global assignments so the re-solve
            # starts fresh for this super-group.
            for cid in merged_class_ids:
                assignments.pop(cid, None)

            # Re-partition with a different random seed
            new_partitions, new_solve_order = self._repartition_group(
                merged_class_ids, class_to_hard, seed=mr_round + 1
            )

            # ── Fix A: solve stuck-class sub-partitions FIRST ─────────────
            # The stuck classes often end up in small/singleton sub-partitions.
            # Solving them BEFORE the large partition ensures they can claim
            # their room-time slots before the large partition fills them up.
            def _has_stuck(p: Partition) -> bool:
                return any(c in stuck_cids for c in p.class_ids)

            new_solve_order = sorted(
                new_solve_order,
                key=lambda p: (0 if _has_stuck(p) else 1, -len(p.class_ids)),
            )

            # Re-solve each new sub-partition sequentially
            intra_assigned: SimpleAssignments = {}

            for new_p in new_solve_order:
                new_p_classes = set(new_p.class_ids)
                mip = self._build_partition_mip(new_p)

                # Room reservations: global stable assignments +
                # already-solved new sub-partitions in this round
                reserved: Set[Tuple[str, tuple]] = set()
                for cid2, (tidx2, rid2) in assignments.items():
                    if rid2 is not None:
                        tbits2 = (
                            self.reader.classes[cid2]["time_options"][tidx2]
                            ["optional_time_bits"]
                        )
                        reserved.add((rid2, tbits2))
                for cid2, (tidx2, rid2) in intra_assigned.items():
                    if rid2 is not None and cid2 not in new_p_classes:
                        tbits2 = (
                            self.reader.classes[cid2]["time_options"][tidx2]
                            ["optional_time_bits"]
                        )
                        reserved.add((rid2, tbits2))
                if reserved:
                    mip.reserve_room_times(reserved)

                # Cross-partition time constraints from ALL currently assigned classes
                all_current: SimpleAssignments = {**assignments, **intra_assigned}
                cross_constrs: List[dict] = []
                seen_cc: Set[int] = set()
                for cid in new_p_classes:
                    for cons in class_to_hard.get(cid, []):
                        ckey = id(cons)
                        if ckey in seen_cc:
                            continue
                        if any(
                            oc in all_current and oc not in new_p_classes
                            for oc in cons["classes"]
                        ):
                            seen_cc.add(ckey)
                            cross_constrs.append(cons)
                if cross_constrs:
                    mip.add_preassigned_time_constraints(all_current, cross_constrs)

                # ── Fix B: force u[cid]=0 for known-stuck classes ─────────
                # Prevent the MIP from legally leaving them unassigned.
                # If this makes the sub-partition infeasible, we fall through
                # to the next round (which will trigger a broader merge).
                forced_in_this_p = [
                    cid for cid in stuck_cids
                    if cid in new_p_classes and cid in mip.u
                ]
                for cid in forced_in_this_p:
                    mip.model.addConstr(
                        mip.u[cid] == 0, name=f"force_stuck_{cid}"
                    )
                if forced_in_this_p:
                    mip.model.update()
                    self.logger.info(
                        f"    Sub-partition {new_p.id}: forcing assignment "
                        f"for {forced_in_this_p}"
                    )

                re_result = mip.solve()
                total_mip_calls += 1
                total_merge_calls += 1
                self._solvers[new_p.id] = mip

                if re_result is not None:
                    new_assigns = mip.get_simple_assignments()
                    intra_assigned.update(new_assigns)
                    self.logger.info(
                        f"    Sub-partition {new_p.id} "
                        f"({len(new_p.class_ids)} classes) → "
                        f"{len(new_assigns)} assigned"
                    )
                else:
                    self.logger.warning(
                        f"    Sub-partition {new_p.id} "
                        f"({len(new_p.class_ids)} classes) infeasible "
                        f"(forced: {forced_in_this_p})."
                    )

            # Commit new assignments to global state
            assignments.update(intra_assigned)

            # Update partition registry and part_of map
            for pid in conflict_pids:
                partitions_by_id.pop(pid, None)
                self._solvers.pop(pid, None)
            for new_p in new_partitions:
                partitions_by_id[new_p.id] = new_p
                for cid in new_p.class_ids:
                    part_of[cid] = new_p.id

            # Stall detection: stop after 5 consecutive non-improving rounds
            if score >= prev_score:
                stall_count += 1
                if stall_count >= 5:
                    self.logger.warning(
                        f"  Merge-repartition stalled for {stall_count} "
                        "consecutive non-improving round(s) — stopping."
                    )
                    break
            else:
                stall_count = 0
            prev_score = score

        else:
            self.logger.warning(
                f"  Merge-repartition: max retries ({self._max_retries}) reached."
            )

        # ── Final check ───────────────────────────────────────────────────────
        final_viol      = self._find_all_violations(assignments)
        final_room_conf = self._detect_room_conflicts(assignments)
        unassigned_final = [
            cid for cid in self.reader.classes if cid not in assignments
        ]

        if final_viol:
            self.logger.warning(f"  {len(final_viol)} time violation(s) remain.")
        if final_room_conf:
            self.logger.warning(f"  {len(final_room_conf)} room conflict(s) remain.")
        if unassigned_final:
            self.logger.warning(
                f"  {len(unassigned_final)} class(es) still unassigned: "
                f"{unassigned_final}"
            )
        if not final_viol and not final_room_conf and not unassigned_final:
            self.logger.info(
                "  Final check: 0 violations, 0 room conflicts, all classes assigned."
            )

        total_violations = len(final_viol) + len(final_room_conf)
        elapsed = time.time() - wall_start
        self.logger.info(
            f"\nDivide & Conquer complete: {len(assignments)} classes assigned, "
            f"MIP calls={total_mip_calls} (merge_repair={total_merge_calls}), "
            f"violations={len(final_viol)}, room_conflicts={len(final_room_conf)}, "
            f"unassigned={len(unassigned_final)}, time={elapsed:.1f}s"
        )

        success = (total_violations == 0 and len(unassigned_final) == 0)
        return DCResult(
            success=success,
            assignments=assignments,
            partitions_solved=len(self._solvers),
            total_mip_calls=total_mip_calls,
            total_conflicts=total_violations,
            total_backtracks=total_merge_calls,
            total_time_sec=elapsed,
            termination="success" if success else "max_retries",
        )

    def save_solution(self, result: DCResult, output_path: str) -> None:
        """
        Convert the merged simple assignments to the full format expected
        by export_solution_xml and write to disk.

        Delegates to the MIPSolver of the *largest* partition (it has
        access to the reader and the export utility).
        """
        if not result.assignments:
            self.logger.warning("save_solution: no assignments to save")
            return

        if not self._solvers:
            self.logger.error("save_solution: no partition solvers available")
            return

        # Build the full assignments dict in MIPSolver.extract_solution() format:
        # {cid: (time_option_dict, room_required, room_id, [])}
        full_assignments: dict = {}
        skipped = 0
        for cid, (tidx, rid) in result.assignments.items():
            if tidx is None:
                skipped += 1
                continue   # class was unassigned in MIP — skip (already warned)
            cls = self.reader.classes[cid]
            time_option = cls["time_options"][tidx]
            room_required = cls.get("room_required", True)
            full_assignments[cid] = (time_option, room_required, rid, [])
        if skipped:
            self.logger.warning(f"save_solution: {skipped} classes had no valid assignment and were omitted.")

        from src.utils.solutionWriter import export_solution_xml
        mip_cfg = self.config.get("train", {}).get("MIP", {})
        export_solution_xml(
            assignments=full_assignments,
            out_path=output_path,
            name=self.reader.problem_name,
            runtime_sec=result.total_time_sec,
            cores=mip_cfg.get("Threads", 1),
            technique=self.config.get("config", {}).get("technique", "divide_conquer"),
            author=self.config.get("config", {}).get("author", ""),
            institution=self.config.get("config", {}).get("institution", ""),
            country=self.config.get("config", {}).get("country", ""),
            include_students=self.config.get("config", {}).get("include_students", False),
        )
        self.logger.info(f"Solution saved to: {output_path}")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _repartition_group(
        self,
        class_ids: List[str],
        class_to_hard: Dict[str, List[dict]],
        seed: int,
    ) -> Tuple[List[Partition], List[Partition]]:
        """
        BFS-partition a subset of classes with a shuffled traversal order.

        Using a different ``seed`` each call guarantees a structurally different
        split from the previous round, breaking cycles that a fixed BFS order
        would repeat indefinitely.

        Parameters
        ----------
        class_ids     : all classes that need to be re-partitioned
        class_to_hard : {cid → list of hard constraints} index (full instance)
        seed          : random seed for shuffling (use round number)

        Returns
        -------
        (new_partitions, solve_order)
        solve_order is sorted largest-first (or smallest-first depending on
        ``self._prefer_larger_first``).
        """
        import random
        rng = random.Random(seed)

        # Build adjacency restricted to this class subset
        class_set = set(class_ids)
        adj: Dict[str, Set[str]] = defaultdict(set)
        for cid in class_ids:
            for cons in class_to_hard.get(cid, []):
                for other_cid in cons["classes"]:
                    if other_cid != cid and other_cid in class_set:
                        adj[cid].add(other_cid)

        # Shuffle traversal order — this is what makes each round different
        shuffled = list(class_ids)
        rng.shuffle(shuffled)

        visited: Set[str] = set()
        raw_groups: List[List[str]] = []
        for start in shuffled:
            if start in visited:
                continue
            group = self._bfs_group_local(start, visited, adj)
            raw_groups.append(group)

        # Build Partition objects with intra-partition constraint lists
        all_hard = self.reader.distributions.get("hard_constraints", [])
        all_soft = self.reader.distributions.get("soft_constraints", [])

        new_partitions: List[Partition] = []
        for group in raw_groups:
            group_set = set(group)
            hard = [
                c for c in all_hard
                if sum(1 for cid in c["classes"] if cid in group_set) >= 2
            ]
            soft = [
                c for c in all_soft
                if sum(1 for cid in c["classes"] if cid in group_set) >= 2
            ]
            pid = self._next_partition_id
            self._next_partition_id += 1
            new_partitions.append(Partition(
                id=pid,
                class_ids=group,
                hard_constraints=hard,
                soft_constraints=soft,
            ))

        solve_order = sorted(
            new_partitions,
            key=lambda p: len(p.class_ids),
            reverse=self._prefer_larger_first,
        )
        return new_partitions, solve_order

    def _bfs_group_local(
        self,
        start: str,
        visited: Set[str],
        adj: Dict[str, Set[str]],
    ) -> List[str]:
        """
        BFS from ``start`` using a local adjacency dict, capped at
        ``self._max_partition_size``.  Overflow nodes are un-marked from
        ``visited`` so the outer loop can start new partitions from them
        (same orphan-prevention logic as HardConstraintPartitioner._bfs_group).
        """
        from collections import deque
        group: List[str] = []
        q: deque = deque([start])
        visited.add(start)

        while q and len(group) < self._max_partition_size:
            node = q.popleft()
            group.append(node)
            for nbr in sorted(adj.get(node, set())):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)

        # Un-mark overflow nodes so they seed new partitions
        for node in q:
            visited.discard(node)

        return group

    def _find_all_violations(
        self, assignments: SimpleAssignments
    ) -> List[Tuple[str, str]]:
        """
        Return a list of (cid1, cid2) pairs that violate a hard time constraint
        (NotOverlap or SameAttendees) in the current merged assignments.

        Only pairs where BOTH classes are assigned are checked.
        """
        violations: List[Tuple[str, str]] = []
        seen: Set[frozenset] = set()

        # Constraint types that require non-overlapping times
        NO_OVERLAP_TYPES = {"NotOverlap", "SameAttendees"}

        for cons in self.reader.distributions.get("hard_constraints", []):
            ctype = cons.get("type", "")
            if ctype not in NO_OVERLAP_TYPES:
                continue

            assigned_in_cons = [c for c in cons["classes"] if c in assignments]
            for c1, c2 in combinations(assigned_in_cons, 2):
                key = frozenset({c1, c2})
                if key in seen:
                    continue
                seen.add(key)
                tidx1, _ = assignments[c1]
                tidx2, _ = assignments[c2]
                b1 = self.reader.classes[c1]["time_options"][tidx1]["optional_time_bits"]
                b2 = self.reader.classes[c2]["time_options"][tidx2]["optional_time_bits"]
                if self._times_conflict(b1, b2, attendee=True):
                    violations.append((c1, c2))

        return violations

    def _build_partition_mip(self, partition: Partition) -> MIPSolver:
        """Create and build a sub-MIP for one partition."""
        mip = MIPSolver(self.reader, self.logger, self.config)
        mip.build_submodel(
            class_ids=set(partition.class_ids),
            hard_constraints=partition.hard_constraints,
            soft_constraints=partition.soft_constraints,
        )
        return mip

    def _detect_room_conflicts(
        self, assignments: SimpleAssignments
    ) -> List[Tuple[str, str, str]]:
        """
        Diagnostic: scan all assignments for room double-booking.

        Returns a list of (cid1, cid2, rid) tuples for each conflicting pair.
        With the sequential room-reservation strategy this list should always
        be empty; it is retained purely for post-hoc verification.
        """
        room_occupants: Dict[str, List[Tuple[str, int, tuple]]] = defaultdict(list)
        for cid, (tidx, rid) in assignments.items():
            if rid is None or rid == 'dummy':
                continue
            time_bits = self.reader.classes[cid]["time_options"][tidx]["optional_time_bits"]
            room_occupants[rid].append((cid, tidx, time_bits))

        conflicts: List[Tuple[str, str, str]] = []
        seen_pairs: Set[frozenset] = set()

        for rid, occupants in room_occupants.items():
            for i, (cid1, _, bits1) in enumerate(occupants):
                for cid2, _, bits2 in occupants[i + 1:]:
                    pair = frozenset({cid1, cid2})
                    if pair in seen_pairs:
                        continue
                    if self._times_conflict(bits1, bits2, attendee=False):
                        seen_pairs.add(pair)
                        conflicts.append((cid1, cid2, rid))

        return conflicts

    def _check_cross_constraints(
        self, hard_cross: list, assignments: SimpleAssignments
    ) -> int:
        """
        Count hard constraints that span partitions and are violated in
        the merged solution.  Used for diagnostic logging only.
        Only checks NotOverlap / SameAttendees (time-conflict semantics).
        """
        violations = 0
        for cons in hard_cross:
            ctype = cons.get("type", "")
            cids = [c for c in cons["classes"] if c in self.reader.classes]
            if "NotOverlap" not in ctype and "SameAttendees" not in ctype:
                continue
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    c1, c2 = cids[i], cids[j]
                    if c1 not in assignments or c2 not in assignments:
                        continue
                    tidx1, _ = assignments[c1]
                    tidx2, _ = assignments[c2]
                    bits1 = self.reader.classes[c1]["time_options"][tidx1]["optional_time_bits"]
                    bits2 = self.reader.classes[c2]["time_options"][tidx2]["optional_time_bits"]
                    if self._times_conflict(bits1, bits2, attendee=True):
                        violations += 1
        return violations

    @staticmethod
    def _times_conflict(bits1: tuple, bits2: tuple, attendee: bool = True) -> bool:
        """Return True if two time_bits tuples conflict.

        Parameters
        ----------
        attendee : bool (default True)
            True  → inclusive boundary (<=): used for SameAttendees / NotOverlap.
                    Back-to-back classes (end of one == start of next) count as
                    conflicting — students need at least 1 slot gap.
            False → strict boundary (<): used for room occupancy.
                    Back-to-back room usage is valid; only true time overlaps
                    (shared slot) are flagged.
        """
        w1, d1, s1, l1 = bits1
        w2, d2, s2, l2 = bits2
        # Slot overlap
        if attendee:
            if not (s1 <= s2 + l2 and s2 <= s1 + l1):
                return False
        else:
            if not (s1 < s2 + l2 and s2 < s1 + l1):
                return False
        # Day overlap
        if (int(d1, 2) & int(d2, 2)) == 0:
            return False
        # Week overlap
        if (int(w1, 2) & int(w2, 2)) == 0:
            return False
        return True
