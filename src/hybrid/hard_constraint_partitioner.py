"""
hard_constraint_partitioner.py
================================
BFS-based graph partitioning of classes using the hard (and soft)
constraint graph.

Why BFS instead of Union-Find
------------------------------
Union-Find merges *all* transitively connected classes into a single
component.  For lums-spr18 this produces a 354-class mega-component that
is still too large for MIP.

BFS partitioning follows the same edges but caps each group at
`max_partition_size`.  Neighbours are enqueued (and marked visited) as
soon as they're discovered, so constraint pairs A-B tend to land in the
same group as long as both fit before the cap is hit.

When the cap is reached and the BFS queue still has nodes, those nodes
are *un-marked* from `visited` so the outer loop can seed new partitions
from them.  This avoids the "orphan" bug (nodes marked visited but never
placed in any partition) at the cost of some cross-partition hard
constraints for the mega-component.  On lums-spr18 this yields ~36 %
cross-partition hard constraints, which are enforced during sequential
solving by ``DivideConquerSolver.add_preassigned_time_constraints()``.

Public types
------------
  Partition               — one BFS group: class_ids + intra constraints
  HardConstraintPartitioner — main class
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Partition:
    """One BFS partition of the class-constraint graph."""
    id: int
    class_ids: List[str]           # all classes in this partition
    hard_constraints: List[dict]   # hard constraints fully within this partition
    soft_constraints: List[dict]   # soft constraints fully within this partition


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HardConstraintPartitioner:
    """
    Partition a timetabling instance's classes into groups using BFS on the
    constraint graph so that each group has at most `max_partition_size`
    classes and (ideally) all hard constraints are intra-partition.

    Parameters
    ----------
    reader              : PSTTReader instance
    max_partition_size  : upper bound on classes per partition (default 50)
    use_soft_edges      : if True, soft-constraint edges are also used to
                          keep soft-constraint class pairs together
                          (default False — only hard edges drive the BFS)
    """

    def __init__(
        self,
        reader,
        max_partition_size: int = 50,
        use_soft_edges: bool = False,
    ):
        self.reader = reader
        self.max_partition_size = max_partition_size
        self.use_soft_edges = use_soft_edges

        # Adjacency: {cid -> set of neighbour cids sharing a constraint}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._partitions: Optional[List[Partition]] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def partition(self) -> List[Partition]:
        """
        Run BFS partitioning and return the list of Partition objects.
        Results are cached; repeated calls are a no-op.
        """
        if self._partitions is not None:
            return self._partitions

        self._build_adjacency()

        visited: Set[str] = set()
        raw_groups: List[List[str]] = []

        # Use insertion order (XML file order): classes that share constraints
        # tend to appear close together in the file, so dict order keeps
        # constraint pairs within the same BFS group far more often than
        # alphabetical order.  Python 3.7+ guarantees insertion-order dicts.
        for cid in self.reader.classes.keys():
            if cid in visited:
                continue
            group = self._bfs_group(cid, visited)
            raw_groups.append(group)

        # Build Partition objects with intra-partition constraint lists
        all_hard = self.reader.distributions.get("hard_constraints", [])
        all_soft = self.reader.distributions.get("soft_constraints", [])

        partitions: List[Partition] = []
        for pid, group in enumerate(raw_groups):
            group_set = set(group)
            # Include a constraint whenever ≥2 of its classes are in this partition.
            # The sub-MIP's _add_single_distribution_constraint already filters to
            # in-model classes, so cross-partition class IDs are ignored at solve time.
            # This avoids silently dropping constraints whose in-partition pairs need
            # to be mutually constrained even when one class lives in another partition.
            hard = [
                c for c in all_hard
                if sum(1 for cid in c["classes"] if cid in group_set) >= 2
            ]
            soft = [
                c for c in all_soft
                if sum(1 for cid in c["classes"] if cid in group_set) >= 2
            ]
            partitions.append(Partition(
                id=pid,
                class_ids=group,
                hard_constraints=hard,
                soft_constraints=soft,
            ))

        self._partitions = partitions
        return partitions

    def get_cross_constraints(self) -> Tuple[List[dict], List[dict]]:
        """
        Return (hard_cross, soft_cross): constraints whose classes span
        more than one partition.  These will need to be checked externally
        after merging partition solutions.
        """
        partitions = self.partition()
        part_of: Dict[str, int] = {}
        for p in partitions:
            for c in p.class_ids:
                part_of[c] = p.id

        all_hard = self.reader.distributions.get("hard_constraints", [])
        all_soft = self.reader.distributions.get("soft_constraints", [])

        # Only consider class IDs actually present in reader.classes
        # (some instances have constraint entries for classes that were removed)
        def _is_cross(cons: dict) -> bool:
            known = {part_of[cid] for cid in cons["classes"] if cid in part_of}
            return len(known) > 1

        hard_cross = [c for c in all_hard if _is_cross(c)]
        soft_cross = [c for c in all_soft if _is_cross(c)]
        return hard_cross, soft_cross

    def part_of_map(self) -> Dict[str, int]:
        """Return {cid: partition_id} for all classes."""
        result: Dict[str, int] = {}
        for p in self.partition():
            for c in p.class_ids:
                result[c] = p.id
        return result

    def summary(self) -> str:
        partitions = self.partition()
        hard_cross, soft_cross = self.get_cross_constraints()
        sizes = sorted([len(p.class_ids) for p in partitions], reverse=True)
        total_hard = len(self.reader.distributions.get("hard_constraints", []))
        total_soft = len(self.reader.distributions.get("soft_constraints", []))
        lines = [
            "=== Hard Constraint Partitioner ===",
            f"  Classes        : {len(self.reader.classes)}",
            f"  Partitions     : {len(partitions)}",
            f"  Sizes (top-10) : {sizes[:10]}{'...' if len(sizes) > 10 else ''}",
            f"  Hard cross     : {len(hard_cross)}/{total_hard} "
            f"({100*len(hard_cross)/max(total_hard,1):.1f}%)  "
            f"[enforced via sequential time-reservation]",
            f"  Soft cross     : {len(soft_cross)}/{total_soft} "
            f"({100*len(soft_cross)/max(total_soft,1):.1f}%)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_adjacency(self) -> None:
        """Populate self._adjacency from hard (and optionally soft) constraints."""
        sources = [self.reader.distributions.get("hard_constraints", [])]
        if self.use_soft_edges:
            sources.append(self.reader.distributions.get("soft_constraints", []))

        for cons_list in sources:
            for cons in cons_list:
                cids = cons["classes"]
                for i in range(len(cids)):
                    for j in range(i + 1, len(cids)):
                        self._adjacency[cids[i]].add(cids[j])
                        self._adjacency[cids[j]].add(cids[i])

    def _bfs_group(self, start: str, visited: Set[str]) -> List[str]:
        """
        BFS from `start`, collecting up to `max_partition_size` nodes.

        Neighbours are marked visited *when enqueued* to prevent duplicate
        enqueuing and to keep constraint pairs A-B in the same group as long
        as they both fit within the size limit.

        When the group hits `max_partition_size` and the BFS queue still has
        items, those items are *un-marked* from `visited` so that the outer
        loop can start new partitions from them.  This prevents the "orphan"
        bug where nodes are marked visited but never placed in any partition.
        """
        group: List[str] = []
        q: deque = deque([start])
        visited.add(start)

        while q and len(group) < self.max_partition_size:
            node = q.popleft()
            group.append(node)
            # Sort neighbours for determinism
            for nbr in sorted(self._adjacency.get(node, set())):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)

        # Nodes still in the queue were pre-marked visited but never processed.
        # Un-mark them so the outer loop can start new partitions from them.
        for node in q:
            visited.discard(node)

        return group
