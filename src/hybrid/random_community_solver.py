"""
random_community_solver.py
===========================
Phase 2 of the two-layer hybrid solver.

Responsibilities:
  1. Enumerate the valid (tidx, rid) action space for every class in a
     community (respecting room availability, room_required flag).
  2. Randomly sample a complete assignment for a community that satisfies
     all its special constraints internally.
  3. Track no-good cuts — community assignments that were already tried and
     judged infeasible by the MIP layer — to avoid resampling them.
  4. Apply a constraint-ordered sampling strategy: assign classes with the
     fewest valid (time, room) options first to reduce backtracking.

Public API
----------
  RandomCommunitySolver
    .sample(community, max_attempts)  → Assignments | None
    .add_no_good(community_id, assignment)
    .clear_no_goods(community_id)
    .action_space(cid)                → list[(tidx, rid)]
    .stats()                          → dict
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.hybrid.community_solver_base import CommunitySolverBase
from src.hybrid.special_constraint_decomposer import (
    Community,
    SpecialConstraintDecomposer,
    _room_available,
)

# {cid: (tidx, rid_or_None)}
Assignments = Dict[str, Tuple[int, Optional[str]]]

# A no-good is the frozenset of (cid, tidx, rid) tuples that characterise
# one fully-assigned community — immutable and hashable.
NoGood = FrozenSet[Tuple[str, int, Optional[str]]]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode / decode a community assignment as a no-good key
# ─────────────────────────────────────────────────────────────────────────────
def _assignment_to_nogood(assignment: Assignments) -> NoGood:
    return frozenset((cid, tidx, rid) for cid, (tidx, rid) in assignment.items())


# ─────────────────────────────────────────────────────────────────────────────
# RandomCommunitySolver
# ─────────────────────────────────────────────────────────────────────────────
class RandomCommunitySolver(CommunitySolverBase):
    """
    Random sampler for special-constraint community assignments.

    Parameters
    ----------
    reader      : PSTTReader
    decomposer  : SpecialConstraintDecomposer
    seed        : int | None   — RNG seed for reproducibility

    Usage
    -----
    solver = RandomCommunitySolver(reader, decomposer, seed=42)

    for community in communities:
        assignment = solver.sample(community)
        if assignment is None:
            # all options exhausted (or max_attempts reached)
            break
        # pass assignment to MIP
        ...
        if mip_infeasible:
            solver.add_no_good(community.id, assignment)
    """

    # Maximum random attempts per sample() call before giving up
    DEFAULT_MAX_ATTEMPTS = 1000

    def __init__(self, reader, decomposer: SpecialConstraintDecomposer,
                 seed: Optional[int] = None):
        self.reader    = reader
        self.decomposer = decomposer
        self._rng      = random.Random(seed)

        # {community_id: set[NoGood]}
        self._no_goods: Dict[int, Set[NoGood]] = {}

        # Cached action spaces: {cid: [(tidx, rid), ...]}
        self._action_cache: Dict[str, List[Tuple[int, Optional[str]]]] = {}

        # Sampling statistics
        self._stats: Dict[str, int] = {
            "total_sample_calls":    0,
            "total_attempts":        0,
            "special_constraint_rejections": 0,
            "no_good_rejections":    0,
            "successes":             0,
            "exhausted":             0,
        }

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def action_space(self, cid: str) -> List[Tuple[int, Optional[str]]]:
        """
        Return all valid (tidx, rid) pairs for class `cid`.

        Validity = room is available at the given time slot, taking into
        account the room's unavailable periods.  Room-not-required classes
        get rid=None for every time option.

        Results are cached after the first call.
        """
        if cid in self._action_cache:
            return self._action_cache[cid]

        cls  = self.reader.classes[cid]
        options: List[Tuple[int, Optional[str]]] = []
        room_required = cls.get("room_required", True)
        room_opts     = cls.get("room_options", [])

        for tidx, topt in enumerate(cls["time_options"]):
            tb = topt["optional_time_bits"]

            if not room_required or not room_opts:
                options.append((tidx, None))
                continue

            for ropt in room_opts:
                rid       = ropt["id"]
                room_data = self.reader.rooms.get(rid, {})
                if _room_available(room_data, tb):
                    options.append((tidx, rid))

        self._action_cache[cid] = options
        return options

    def sample(
        self,
        community: Community,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> Optional[Assignments]:
        """
        Draw a random assignment for all classes in `community` that:
          1. Uses only valid (tidx, rid) pairs for each class.
          2. Satisfies all special constraints within the community.
          3. Is not in the no-good set for this community.

        Sampling strategy
        -----------------
        Classes are ordered by action-space size (ascending) so the most
        constrained class is assigned first, pruning the search space early.
        Within each class, a uniformly random valid option is chosen.
        This is "random restarts with constraint-ordered assignment" — a
        simple but effective heuristic for small communities.

        Returns
        -------
        Assignments  if a valid, non-no-good assignment is found.
        None         if max_attempts is exhausted or the action space for
                     any class is empty.
        """
        self._stats["total_sample_calls"] += 1

        # Sort classes by action-space size (ascending = most constrained first)
        ordered_cids = self._order_by_constraint(community)

        # Fast-fail: if any class has no valid options, sampling is impossible
        for cid in ordered_cids:
            if not self.action_space(cid):
                self._stats["exhausted"] += 1
                return None

        no_goods = self._no_goods.get(community.id, set())

        for attempt in range(max_attempts):
            self._stats["total_attempts"] += 1

            assignment = self._draw_assignment(ordered_cids)

            # Check special constraints
            if not self.decomposer.check_special_constraints(community, assignment):
                self._stats["special_constraint_rejections"] += 1
                continue

            # Check no-good set
            ng_key = _assignment_to_nogood(assignment)
            if ng_key in no_goods:
                self._stats["no_good_rejections"] += 1
                continue

            self._stats["successes"] += 1
            return assignment

        # max_attempts reached without a valid sample
        self._stats["exhausted"] += 1
        return None

    def add_no_good(self, community_id: int, assignment: Assignments) -> None:
        """
        Register a community assignment as forbidden.

        Call this after the MIP returns INFEASIBLE and
        `get_infeasibility_info` implicates classes in this community.

        Parameters
        ----------
        community_id : Community.id
        assignment   : the full {cid: (tidx, rid)} dict that was tried
        """
        ng = _assignment_to_nogood(assignment)
        self._no_goods.setdefault(community_id, set()).add(ng)

    def clear_no_goods(self, community_id: Optional[int] = None) -> None:
        """
        Remove no-good cuts.

        Parameters
        ----------
        community_id : if given, clear only that community's no-goods;
                       if None, clear all.
        """
        if community_id is None:
            self._no_goods.clear()
        else:
            self._no_goods.pop(community_id, None)

    def no_good_count(self, community_id: int) -> int:
        """Return the number of no-good cuts registered for a community."""
        return len(self._no_goods.get(community_id, set()))

    def stats(self) -> dict:
        """Return a copy of the running sampling statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all sampling statistics to zero."""
        for k in self._stats:
            self._stats[k] = 0

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _order_by_constraint(self, community: Community) -> List[str]:
        """
        Return class IDs sorted by action-space size ascending.
        Ties broken by class ID (stable / deterministic).
        """
        return sorted(
            community.class_ids,
            key=lambda cid: (len(self.action_space(cid)), cid),
        )

    def _draw_assignment(self, ordered_cids: List[str]) -> Assignments:
        """
        Draw one (tidx, rid) per class uniformly at random.
        Classes are sampled in `ordered_cids` order.
        """
        assignment: Assignments = {}
        for cid in ordered_cids:
            options = self.action_space(cid)
            tidx, rid = self._rng.choice(options)
            assignment[cid] = (tidx, rid)
        return assignment

    def _community_action_space_size(self, community: Community) -> int:
        """Product of individual action-space sizes (upper bound on combinations)."""
        product = 1
        for cid in community.class_ids:
            n = len(self.action_space(cid))
            if n == 0:
                return 0
            product *= n
        return product
