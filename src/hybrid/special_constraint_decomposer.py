"""
special_constraint_decomposer.py
=================================
Phase 1 of the two-layer hybrid solver.

Responsibilities:
  1. Identify the four "non-binarisable" special constraints
     (MaxDays / MaxDayLoad / MaxBreaks / MaxBlock) from the reader.
  2. Group classes that share these constraints into independent
     *communities* via Union-Find.
  3. Classify every class in the instance as one of:
       - fixed     : exactly one valid (time, room) combination
       - community : member of at least one special-constraint community
       - free      : all other classes (handled directly by MIP)
  4. Provide check_special_constraints() to validate a proposed
     community assignment before passing it to the MIP layer.

Public types
------------
  Community            — one connected component of special constraints
  ClassificationResult — output of classify_classes()
  SpecialConstraintDecomposer — main class
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constant: the four special constraint base-types
# ─────────────────────────────────────────────────────────────────────────────
SPECIAL_TYPES = {"MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock"}

# assignments type alias:  {cid: (tidx, rid_or_None)}
Assignments = Dict[str, Tuple[int, Optional[str]]]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Community:
    """A set of classes linked by at least one special constraint."""
    id: int
    class_ids: Set[str]
    constraints: List[dict]   # all special constraints whose classes ⊆ community


@dataclass
class ClassificationResult:
    fixed: Set[str]              # classes with exactly one valid (time, room)
    community: Set[str]          # classes belonging to ≥1 community
    free: Set[str]               # remaining classes
    communities: List[Community]


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find (path compression + union by rank)
# ─────────────────────────────────────────────────────────────────────────────
class _UnionFind:
    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def add(self, x: str):
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> Dict[str, Set[str]]:
        """Return {root: {members}}."""
        result: Dict[str, Set[str]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, set()).add(x)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse constraint base-type and parameter
# ─────────────────────────────────────────────────────────────────────────────
def _parse_constraint_type(type_str: str) -> Tuple[str, Optional[str]]:
    """
    "MaxDays(3)"   → ("MaxDays", "3")
    "MaxBreaks(1,6)" → ("MaxBreaks", "1,6")
    "NotOverlap"   → ("NotOverlap", None)
    """
    m = re.match(r"^(\w+)\(([^)]+)\)$", type_str)
    if m:
        return m.group(1), m.group(2)
    return type_str, None


def _is_special(type_str: str) -> bool:
    base, _ = _parse_constraint_type(type_str)
    return base in SPECIAL_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Time-slot helpers (work directly on time_bits tuples)
# ─────────────────────────────────────────────────────────────────────────────
def _active_on_day(time_bits, week: int, day: int) -> bool:
    """True if this time option is scheduled during (week, day)."""
    weeks_bits, days_bits, _, _ = time_bits
    return weeks_bits[week] == '1' and days_bits[day] == '1'


def _merge_slots(slots: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """
    Merge adjacent/nearby [start, length] slots.
    Two slots merge if  start1 + length1 + gap >= start2.
    Returns list of merged (start, length) tuples.
    """
    if not slots:
        return []
    sorted_slots = sorted(slots, key=lambda s: s[0])
    merged = [list(sorted_slots[0])]
    for start, length in sorted_slots[1:]:
        prev_start, prev_len = merged[-1]
        if prev_start + prev_len + gap >= start:
            merged[-1][1] = max(prev_start + prev_len, start + length) - prev_start
        else:
            merged.append([start, length])
    return [tuple(s) for s in merged]


# ─────────────────────────────────────────────────────────────────────────────
# Room availability check (mirrors solver._is_room_available)
# ─────────────────────────────────────────────────────────────────────────────
def _room_available(room_data: dict, time_bits: tuple) -> bool:
    """Return True if the room is available for the given time_bits."""
    unavailables = room_data.get('unavailables_bits', [])
    if not unavailables:
        return True
    w_bits, d_bits, start, length = time_bits
    for unavail in unavailables:
        uw, ud, us, ul = unavail
        if uw is None or ud is None or us is None or ul is None:
            continue
        # check week overlap
        if not (int(w_bits, 2) & int(uw, 2)):
            continue
        # check day overlap
        if not (int(d_bits, 2) & int(ud, 2)):
            continue
        # check slot overlap
        if start < us + ul and us < start + length:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Special constraint checkers
# Each takes:
#   class_time_bits : {cid: (weeks_bits, days_bits, start, length)}
#   param           : the parameter string from the constraint type
#   nrWeeks, nrDays : problem dimensions
# Returns True if the assignment SATISFIES the constraint.
# ─────────────────────────────────────────────────────────────────────────────

def _check_max_days(class_time_bits: Dict[str, tuple], param: str,
                    nrWeeks: int, nrDays: int) -> bool:
    """All classes together must span ≤ D distinct days (across all weeks)."""
    D = int(param)
    days_union = 0
    for tb in class_time_bits.values():
        days_union |= int(tb[1], 2)
    return bin(days_union).count('1') <= D


def _check_max_day_load(class_time_bits: Dict[str, tuple], param: str,
                        nrWeeks: int, nrDays: int) -> bool:
    """For every (week, day), total slot-length of active classes ≤ S."""
    S = int(param)
    time_bits_list = list(class_time_bits.values())
    for w in range(nrWeeks):
        for d in range(nrDays):
            day_load = sum(
                tb[3] for tb in time_bits_list
                if tb[0][w] == '1' and tb[1][d] == '1'
            )
            if day_load > S:
                return False
    return True


def _check_max_breaks(class_time_bits: Dict[str, tuple], param: str,
                      nrWeeks: int, nrDays: int) -> bool:
    """For every (week, day), number of breaks (gaps > S between blocks) ≤ R."""
    R, S = map(int, param.split(","))
    time_bits_list = list(class_time_bits.values())
    for w in range(nrWeeks):
        for d in range(nrDays):
            slots = [
                (tb[2], tb[3]) for tb in time_bits_list
                if tb[0][w] == '1' and tb[1][d] == '1'
            ]
            if len(slots) < 2:
                continue
            merged = _merge_slots(slots, S)
            breaks = len(merged) - 1
            if breaks > R:
                return False
    return True


def _check_max_block(class_time_bits: Dict[str, tuple], param: str,
                     nrWeeks: int, nrDays: int) -> bool:
    """For every (week, day), no merged block (gap ≤ S) spanning ≥2 classes exceeds M slots."""
    M, S = map(int, param.split(","))
    time_bits_list = list(class_time_bits.values())
    for w in range(nrWeeks):
        for d in range(nrDays):
            slots = [
                (tb[2], tb[3]) for tb in time_bits_list
                if tb[0][w] == '1' and tb[1][d] == '1'
            ]
            if len(slots) < 2:
                continue
            # Need to track how many original slots each merged block contains.
            # Sort and merge, tracking member count.
            sorted_slots = sorted(slots, key=lambda s: s[0])
            merged: List[List] = [[sorted_slots[0][0], sorted_slots[0][1], 1]]
            for start, length in sorted_slots[1:]:
                prev = merged[-1]
                if prev[0] + prev[1] + S >= start:
                    prev[1] = max(prev[0] + prev[1], start + length) - prev[0]
                    prev[2] += 1
                else:
                    merged.append([start, length, 1])
            for block_start, block_len, member_count in merged:
                if member_count >= 2 and block_len > M:
                    return False
    return True


_CHECKER_MAP = {
    "MaxDays":    _check_max_days,
    "MaxDayLoad": _check_max_day_load,
    "MaxBreaks":  _check_max_breaks,
    "MaxBlock":   _check_max_block,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class SpecialConstraintDecomposer:
    """
    Decomposes an ITC 2019 instance into special-constraint communities and
    classifies all classes for the two-layer hybrid solver.

    Parameters
    ----------
    reader : PSTTReader
        Parsed problem instance.

    Usage
    -----
    decomposer = SpecialConstraintDecomposer(reader)
    result = decomposer.classify_classes()
    # result.communities  — list of Community objects
    # result.fixed        — classes with a single valid option
    # result.community    — classes in ≥1 community
    # result.free         — remaining classes (pure MIP variables)
    """

    def __init__(self, reader):
        self.reader = reader
        self._communities: Optional[List[Community]] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_communities(self) -> List[Community]:
        """
        Identify special constraints, group classes sharing them via
        Union-Find, and return the list of Community objects.
        Results are cached; call again is a no-op.
        """
        if self._communities is not None:
            return self._communities

        special_constraints = self._collect_special_constraints()
        uf = _UnionFind()

        # Seed union-find with all class IDs
        for cid in self.reader.classes:
            uf.add(cid)

        # Union classes that share a special constraint
        for cons in special_constraints:
            cids = cons["classes"]
            for i in range(1, len(cids)):
                uf.union(cids[0], cids[i])

        # Build communities from the groups that contain ≥2 classes AND
        # are involved in at least one special constraint.
        groups = uf.groups()
        # Determine which roots are actually involved in special constraints
        involved_cids: Set[str] = set()
        for cons in special_constraints:
            involved_cids.update(cons["classes"])

        involved_roots: Set[str] = {uf.find(c) for c in involved_cids}

        communities: List[Community] = []
        for comm_id, (root, members) in enumerate(
            (r, g) for r, g in groups.items() if r in involved_roots
        ):
            # Collect all special constraints whose class-sets are subsets of members
            comm_constraints = [
                c for c in special_constraints
                if set(c["classes"]).issubset(members)
            ]
            communities.append(Community(
                id=comm_id,
                class_ids=members,
                constraints=comm_constraints,
            ))

        self._communities = communities
        return communities

    def classify_classes(self) -> ClassificationResult:
        """
        Classify every class in the instance into fixed / community / free.

        fixed     — classes that have exactly one valid (time, room) pair
                    after filtering by room availability.
        community — classes that belong to at least one special-constraint
                    community (excluding those already in fixed).
        free      — all remaining classes.
        """
        communities = self.build_communities()
        community_cids: Set[str] = set()
        for comm in communities:
            community_cids.update(comm.class_ids)

        fixed: Set[str] = set()
        for cid, cls in self.reader.classes.items():
            valid = self._valid_options(cid, cls)
            if len(valid) == 1:
                fixed.add(cid)

        # community classes that are also "fixed" stay in fixed (they will be
        # handled as fixed assignments, not sampled)
        community_only = community_cids - fixed
        free = set(self.reader.classes.keys()) - fixed - community_only

        return ClassificationResult(
            fixed=fixed,
            community=community_only,
            free=free,
            communities=communities,
        )

    def get_fixed_assignments(self) -> Assignments:
        """
        For every class in the `fixed` set, return its unique valid
        (tidx, rid) assignment.  rid is None for room-not-required classes.
        """
        result: Assignments = {}
        for cid, cls in self.reader.classes.items():
            valid = self._valid_options(cid, cls)
            if len(valid) == 1:
                tidx, rid = valid[0]
                result[cid] = (tidx, rid)
        return result

    def check_special_constraints(
        self,
        community: Community,
        assignments: Assignments,
    ) -> bool:
        """
        Verify that a proposed assignment for a community satisfies **all**
        special constraints in that community.

        Parameters
        ----------
        community   : Community object (from build_communities())
        assignments : {cid: (tidx, rid_or_None)} — must cover all cids in
                      community.class_ids (unassigned cids are skipped with
                      a warning).

        Returns
        -------
        True  if every special constraint in the community is satisfied.
        False if any constraint is violated.
        """
        nrWeeks = self.reader.nrWeeks
        nrDays  = self.reader.nrDays

        for cons in community.constraints:
            base, param = _parse_constraint_type(cons["type"])
            checker = _CHECKER_MAP.get(base)
            if checker is None:
                continue  # non-special constraint — skip

            # Collect time_bits for classes that have an assignment
            class_time_bits: Dict[str, tuple] = {}
            for cid in cons["classes"]:
                if cid not in assignments:
                    continue
                tidx, _ = assignments[cid]
                tb = self.reader.classes[cid]["time_options"][tidx]["optional_time_bits"]
                class_time_bits[cid] = tb

            if not class_time_bits:
                continue  # nothing assigned yet — trivially satisfied

            if not checker(class_time_bits, param, nrWeeks, nrDays):
                return False

        return True

    def summary(self) -> str:
        """Return a human-readable decomposition summary."""
        result = self.classify_classes()
        lines = [
            "=== Special Constraint Decomposer Summary ===",
            f"  Total classes   : {len(self.reader.classes)}",
            f"  Fixed classes   : {len(result.fixed)}",
            f"  Community classes: {len(result.community)}",
            f"  Free classes    : {len(result.free)}",
            f"  Communities     : {len(result.communities)}",
        ]
        for comm in sorted(result.communities, key=lambda c: -len(c.class_ids)):
            type_counts: Dict[str, int] = {}
            for cons in comm.constraints:
                base, _ = _parse_constraint_type(cons["type"])
                type_counts[base] = type_counts.get(base, 0) + 1
            type_str = ", ".join(f"{k}×{v}" for k, v in sorted(type_counts.items()))
            lines.append(
                f"    Community {comm.id:3d}: {len(comm.class_ids):4d} classes, "
                f"{len(comm.constraints):3d} constraints [{type_str}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _collect_special_constraints(self) -> List[dict]:
        """Return all special constraints from both hard and soft lists."""
        special = []
        for cons in self.reader.distributions.get("hard_constraints", []):
            if _is_special(cons["type"]):
                special.append(cons)
        for cons in self.reader.distributions.get("soft_constraints", []):
            if _is_special(cons["type"]):
                special.append(cons)
        return special

    def _valid_options(self, cid: str, cls: dict) -> List[Tuple[int, Optional[str]]]:
        """
        Return all valid (tidx, rid) pairs for a class after filtering
        by room availability.  rid=None for room-not-required classes.
        """
        valid = []
        room_required = cls.get("room_required", True)
        room_opts = cls.get("room_options", [])

        for tidx, topt in enumerate(cls["time_options"]):
            tb = topt["optional_time_bits"]

            if not room_required or not room_opts:
                valid.append((tidx, None))
                continue

            for ropt in room_opts:
                rid = ropt["id"]
                room_data = self.reader.rooms.get(rid, {})
                if _room_available(room_data, tb):
                    valid.append((tidx, rid))

        return valid
