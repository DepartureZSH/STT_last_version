"""
hybrid_solver.py
================
Phase 4 of the two-layer hybrid solver.

Orchestrates the full Layer-1 / Layer-2 / feedback loop:

  Layer 1  — RandomCommunitySolver samples (time, room) assignments for
             every special-constraint community and for single-option
             ("fixed") classes.

  Layer 2  — MIPSolver receives the Layer-1 assignments as hard constraints,
             then solves the remaining free variables.

  Feedback — On INFEASIBLE, MIPSolver.get_infeasibility_info() identifies
             the implicated fixed-class assignments.  Those community
             assignments are added as no-goods so they are not resampled.

Usage
-----
    from src.hybrid.hybrid_solver import HybridSolver
    from src.utils.dataReader import PSTTReader
    import logging, yaml

    reader = PSTTReader('data/source/instances/agh-fal17.xml')
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    hybrid = HybridSolver(reader, config, logger)
    result = hybrid.solve()
    if result:
        hybrid.save_solution(result, 'output/solution.xml')
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.utils.dataReader import PSTTReader
from src.MIP.solver import MIPSolver
from src.hybrid.special_constraint_decomposer import (
    Community,
    ClassificationResult,
    SpecialConstraintDecomposer,
)
from src.hybrid.random_community_solver import RandomCommunitySolver

# {cid: (tidx, rid_or_None)}
Assignments = Dict[str, Tuple[int, Optional[str]]]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HybridResult:
    """Outcome of one HybridSolver.solve() call."""
    success:          bool
    assignments_list: Optional[list]         # MIPSolver.extract_solution() output
    iterations:       int
    total_time_sec:   float
    mip_calls:        int
    infeasible_calls: int
    community_stats:  dict                   # RandomCommunitySolver.stats()
    termination:      str                    # "optimal" | "time_limit" | "max_iter" | "exhausted"


# ─────────────────────────────────────────────────────────────────────────────
# HybridSolver
# ─────────────────────────────────────────────────────────────────────────────
class HybridSolver:
    """
    Two-layer hybrid solver for ITC 2019 timetabling instances.

    Parameters
    ----------
    reader  : PSTTReader
    config  : dict — full config (see config.yaml schema below)
    logger  : logging.Logger | None

    Config schema (relevant keys)
    ------------------------------
    train:
      MIP:
        time_limit:    <int>     # seconds per MIP call
        Threads:       <int>
        MIPGap:        <float>
        PoolSolutions: <int>

    hybrid:
      max_iterations:      <int>    # outer loop cap  (default 100)
      max_attempts_per_community: <int>  # RandomCommunitySolver.sample() cap (default 1000)
      rebuild_mip_every:   <int>    # rebuild full MIP model every N iterations (default 1)
                                    # set >1 to amortise build cost on large instances
      seed:                <int>    # RNG seed (default 42)

    config:
      technique / author / institution / country / include_students
    """

    DEFAULT_HYBRID = {
        "max_iterations":             100,
        "max_attempts_per_community": 1000,
        "rebuild_mip_every":          1,
        "seed":                       42,
    }

    def __init__(self, reader: PSTTReader, config: dict,
                 logger: Optional[logging.Logger] = None):
        self.reader  = reader
        self.config  = config
        self.logger  = logger or logging.getLogger(__name__)

        # Merge hybrid config with defaults
        hybrid_cfg   = {**self.DEFAULT_HYBRID, **config.get("hybrid", {})}
        self._max_iter       = int(hybrid_cfg["max_iterations"])
        self._max_attempts   = int(hybrid_cfg["max_attempts_per_community"])
        self._rebuild_every  = int(hybrid_cfg["rebuild_mip_every"])
        self._seed           = hybrid_cfg["seed"]

        # Layer-1 components (built lazily in solve())
        self._decomposer:  Optional[SpecialConstraintDecomposer] = None
        self._cls_result:  Optional[ClassificationResult]        = None
        self._comm_solver: Optional[RandomCommunitySolver]       = None

        # Layer-2 component (rebuilt as needed)
        self._mip: Optional[MIPSolver] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def solve(self) -> Optional[HybridResult]:
        """
        Run the hybrid solve loop.

        Returns
        -------
        HybridResult with success=True  if a feasible solution is found.
        HybridResult with success=False if all iterations are exhausted.
        """
        wall_start = time.time()
        mip_calls = infeasible_calls = 0

        # ── Phase 1: decompose ───────────────────────────────────────────
        self.logger.info("=== Hybrid Solver: decomposing instance ===")
        self._decomposer  = SpecialConstraintDecomposer(self.reader)
        self._cls_result  = self._decomposer.classify_classes()
        self._comm_solver = RandomCommunitySolver(
            self.reader, self._decomposer, seed=self._seed
        )

        communities     = self._cls_result.communities
        fixed_assigns   = self._decomposer.get_fixed_assignments()

        self.logger.info(
            f"  Classes: {len(self.reader.classes)} total | "
            f"fixed={len(fixed_assigns)} | "
            f"community={len(self._cls_result.community)} | "
            f"free={len(self._cls_result.free)}"
        )
        self.logger.info(f"  Communities: {len(communities)}")

        # Edge case: no communities → pure MIP
        if not communities:
            self.logger.info("  No special-constraint communities — running pure MIP")
            return self._run_pure_mip(fixed_assigns, wall_start)

        # ── Main loop ────────────────────────────────────────────────────
        self._mip = self._build_mip()

        for iteration in range(1, self._max_iter + 1):
            iter_start = time.time()
            self.logger.info(f"\n--- Iteration {iteration}/{self._max_iter} ---")

            # ── Layer 1: sample community assignments ────────────────────
            community_assigns, exhausted_comm = self._sample_all_communities(
                communities
            )
            if exhausted_comm is not None:
                self.logger.warning(
                    f"  Community {exhausted_comm.id} exhausted "
                    f"(no valid assignment found in {self._max_attempts} attempts)"
                )
                return HybridResult(
                    success=False,
                    assignments_list=None,
                    iterations=iteration,
                    total_time_sec=time.time() - wall_start,
                    mip_calls=mip_calls,
                    infeasible_calls=infeasible_calls,
                    community_stats=self._comm_solver.stats(),
                    termination="exhausted",
                )

            # Merge: fixed single-option classes + community assignments
            all_fixed: Assignments = {**fixed_assigns, **community_assigns}
            self.logger.info(
                f"  Layer 1: {len(all_fixed)} classes fixed "
                f"({len(fixed_assigns)} single-option + {len(community_assigns)} community)"
            )

            # ── Layer 2: MIP ─────────────────────────────────────────────
            # Rebuild model if scheduled
            if (iteration - 1) % self._rebuild_every == 0 and iteration > 1:
                self.logger.info("  Rebuilding MIP model")
                self._mip = self._build_mip()

            self._mip.fix_assignments(all_fixed)
            mip_calls += 1

            assignments_list = self._mip.solve()

            iter_time = time.time() - iter_start

            # ── Feasible ─────────────────────────────────────────────────
            if assignments_list is not None:
                self.logger.info(
                    f"  ✓ Feasible solution found at iteration {iteration} "
                    f"(iter_time={iter_time:.1f}s)"
                )
                return HybridResult(
                    success=True,
                    assignments_list=assignments_list,
                    iterations=iteration,
                    total_time_sec=time.time() - wall_start,
                    mip_calls=mip_calls,
                    infeasible_calls=infeasible_calls,
                    community_stats=self._comm_solver.stats(),
                    termination="optimal" if self._mip.model.Status == 2 else "time_limit",
                )

            # ── Infeasible: collect IIS feedback ────────────────────────
            infeasible_calls += 1
            info = self._mip.get_infeasibility_info()
            implicated = info["implicated_cids"]
            self.logger.info(
                f"  ✗ Infeasible — {len(implicated)} implicated classes: {implicated}"
            )

            # Add no-good cuts for communities that contain implicated classes
            self._add_no_goods_from_iis(communities, community_assigns, implicated)

            # Reset fixed constraints for next iteration
            self._mip.reset_fixed()

            # Rebuild model unconditionally after infeasibility to clear
            # any no-good cuts added by extract_solution
            self._mip = self._build_mip()

        # ── Max iterations reached ───────────────────────────────────────
        self.logger.warning(f"Max iterations ({self._max_iter}) reached without solution")
        return HybridResult(
            success=False,
            assignments_list=None,
            iterations=self._max_iter,
            total_time_sec=time.time() - wall_start,
            mip_calls=mip_calls,
            infeasible_calls=infeasible_calls,
            community_stats=self._comm_solver.stats(),
            termination="max_iter",
        )

    def save_solution(self, result: HybridResult, output_path: str) -> None:
        """
        Write the best assignment from a successful HybridResult to XML.
        Delegates to MIPSolver.save_solution().
        """
        if not result.success or result.assignments_list is None:
            self.logger.warning("save_solution: no feasible solution to save")
            return
        if self._mip is None:
            self.logger.error("save_solution: MIP solver not initialised")
            return
        self._mip.save_solution(result.assignments_list[0], output_path, self.config)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_mip(self) -> MIPSolver:
        """Create and build a fresh MIPSolver."""
        mip = MIPSolver(self.reader, self.logger, self.config)
        mip.build_model()
        return mip

    def _run_pure_mip(
        self, fixed_assigns: Assignments, wall_start: float
    ) -> HybridResult:
        """Fallback for instances with no special-constraint communities."""
        mip = self._build_mip()
        if fixed_assigns:
            mip.fix_assignments(fixed_assigns)
        assignments_list = mip.solve()
        success = assignments_list is not None
        return HybridResult(
            success=success,
            assignments_list=assignments_list,
            iterations=1,
            total_time_sec=time.time() - wall_start,
            mip_calls=1,
            infeasible_calls=0 if success else 1,
            community_stats={},
            termination="optimal" if success else "infeasible",
        )

    def _sample_all_communities(
        self, communities: List[Community]
    ) -> Tuple[Assignments, Optional[Community]]:
        """
        Sample an assignment for every community.

        Returns
        -------
        (merged_assignments, None)       on success
        ({},                community)   if sampling for `community` is exhausted
        """
        merged: Assignments = {}
        for comm in communities:
            assignment = self._comm_solver.sample(
                comm, max_attempts=self._max_attempts
            )
            if assignment is None:
                return {}, comm
            merged.update(assignment)
        return merged, None

    def _add_no_goods_from_iis(
        self,
        communities:       List[Community],
        community_assigns: Assignments,
        implicated_cids:   Set[str],
    ) -> None:
        """
        For each community that contains at least one IIS-implicated class,
        register the community's current assignment as a no-good.
        """
        for comm in communities:
            if comm.class_ids & implicated_cids:
                comm_assignment = {
                    cid: community_assigns[cid]
                    for cid in comm.class_ids
                    if cid in community_assigns
                }
                self._comm_solver.add_no_good(comm.id, comm_assignment)
                self.logger.info(
                    f"  No-good added for community {comm.id} "
                    f"(now {self._comm_solver.no_good_count(comm.id)} no-goods)"
                )
