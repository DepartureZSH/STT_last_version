"""
marl_community_solver.py
========================
Phase 5 — MARL-based community assignment solver (stub / interface).

This file provides the class skeleton that Phase 5 will flesh out.
All public methods raise NotImplementedError; the signatures and
docstrings define the full contract expected by HybridSolver.

Architecture sketch
-------------------
Each Community becomes a cooperative multi-agent environment:
  - Agent  = one class within the community
  - Action = (tidx, rid) index pair  (discrete, from action_space)
  - State  = current partial assignment of the community
  - Reward = per-agent scalar (see MARLCommunitySolverBase.update_reward)

Suggested algorithm: QMIX or independent Q-learning with shared replay
buffer, trained offline against the special-constraint checker and refined
online via MIP feedback.

To implement Phase 5
--------------------
1. Choose a MARL framework (e.g., RLlib, TorchMARL, custom PyTorch).
2. Implement each NotImplementedError method below.
3. Update HybridSolver to accept a CommunitySolverBase and pass a
   MARLCommunitySolver instance instead of RandomCommunitySolver.
4. Run tests/phase5_marl_solver_test.py — all SKIP marks will lift once
   is_trained returns True.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.hybrid.community_solver_base import MARLCommunitySolverBase
from src.hybrid.special_constraint_decomposer import (
    Community,
    SpecialConstraintDecomposer,
    _room_available,
)

Assignments = Dict[str, Tuple[int, Optional[str]]]
NoGood = FrozenSet[Tuple[str, int, Optional[str]]]


class MARLCommunitySolver(MARLCommunitySolverBase):
    """
    MARL-based community assignment solver (Phase 5 stub).

    Parameters
    ----------
    reader      : PSTTReader
    decomposer  : SpecialConstraintDecomposer
    device      : "cpu" | "cuda" — target device for neural networks
    lr          : float — learning rate
    gamma       : float — discount factor
    seed        : int | None

    All methods raise NotImplementedError until Phase 5 is implemented.
    The action_space, add_no_good, clear_no_goods, no_good_count, stats,
    and reset_stats methods are fully implemented (copied logic from
    RandomCommunitySolver) so that interface-level tests pass immediately.
    """

    def __init__(
        self,
        reader,
        decomposer: SpecialConstraintDecomposer,
        *,
        device: str = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        self.reader     = reader
        self.decomposer = decomposer
        self.device     = device
        self.lr         = lr
        self.gamma      = gamma
        self._seed      = seed

        # No-good tracking (identical to RandomCommunitySolver)
        self._no_goods: Dict[int, Set[NoGood]] = {}

        # Action space cache
        self._action_cache: Dict[str, List[Tuple[int, Optional[str]]]] = {}

        # Statistics
        self._stats: Dict[str, object] = {
            "total_sample_calls":            0,
            "total_attempts":                0,
            "special_constraint_rejections": 0,
            "no_good_rejections":            0,
            "successes":                     0,
            "exhausted":                     0,
            # MARL-specific
            "train_episodes":                0,
            "mean_reward":                   None,
            "update_reward_calls":           0,
        }

        self._is_trained = False

    # ------------------------------------------------------------------ #
    # Fully implemented: shared with RandomCommunitySolver                #
    # ------------------------------------------------------------------ #

    def action_space(self, cid: str) -> List[Tuple[int, Optional[str]]]:
        """Enumerate valid (tidx, rid) pairs for class `cid` (cached)."""
        if cid in self._action_cache:
            return self._action_cache[cid]

        cls           = self.reader.classes[cid]
        options: List[Tuple[int, Optional[str]]] = []
        room_required = cls.get("room_required", True)
        room_opts     = cls.get("room_options", [])

        for tidx, topt in enumerate(cls["time_options"]):
            tb = topt["optional_time_bits"]
            if not room_required or not room_opts:
                options.append((tidx, None))
                continue
            for ropt in room_opts:
                rid = ropt["id"]
                if _room_available(self.reader.rooms.get(rid, {}), tb):
                    options.append((tidx, rid))

        self._action_cache[cid] = options
        return options

    def add_no_good(self, community_id: int, assignment: Assignments) -> None:
        """Register a forbidden assignment for community `community_id`."""
        ng = frozenset((cid, tidx, rid) for cid, (tidx, rid) in assignment.items())
        self._no_goods.setdefault(community_id, set()).add(ng)

    def clear_no_goods(self, community_id: Optional[int] = None) -> None:
        """Remove no-good cuts (all or for a specific community)."""
        if community_id is None:
            self._no_goods.clear()
        else:
            self._no_goods.pop(community_id, None)

    def no_good_count(self, community_id: int) -> int:
        """Return the number of no-good cuts for community `community_id`."""
        return len(self._no_goods.get(community_id, set()))

    def stats(self) -> dict:
        """Return a copy of current statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all statistics to zero / None."""
        for k in self._stats:
            self._stats[k] = 0 if k != "mean_reward" else None

    @property
    def is_trained(self) -> bool:
        """True once train() or load_model() has been called successfully."""
        return self._is_trained

    # ------------------------------------------------------------------ #
    # Phase 5 stubs — implement these                                     #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        community: Community,
        max_attempts: int = 1000,
    ) -> Optional[Assignments]:
        """
        Use trained policy networks to select (tidx, rid) for each class.

        Falls back to random sampling if not yet trained.

        TODO (Phase 5):
          - Build community observation tensor from current state.
          - Forward pass through each agent's Q-network / policy.
          - Apply epsilon-greedy or softmax action selection.
          - Check special constraints; reject or accept.
          - Check no-good set; reject if seen.
          - Return assignment or None if max_attempts reached.
        """
        raise NotImplementedError(
            "MARLCommunitySolver.sample() is not yet implemented. "
            "Train the model first via train(), or use RandomCommunitySolver."
        )

    def train(
        self,
        communities: list,
        n_episodes: int,
        *,
        verbose: bool = False,
    ) -> dict:
        """
        Train agent policies for all communities against the special-constraint
        environment.

        TODO (Phase 5):
          - Build one MARL environment per community.
          - Run n_episodes of experience collection and gradient updates.
          - Track mean reward, win rate, constraint satisfaction rate.
          - Set self._is_trained = True on completion.
          - Return training summary dict.
        """
        raise NotImplementedError(
            "MARLCommunitySolver.train() is not yet implemented (Phase 5)."
        )

    def update_reward(
        self,
        community_id:    int,
        assignment:      Assignments,
        implicated_cids: Set[str],
        mip_feasible:    bool,
    ) -> None:
        """
        Deliver online MIP-feedback reward to the community's agents.

        TODO (Phase 5):
          - Reconstruct which agent took which action from `assignment`.
          - Compute per-agent reward:
              mip_feasible=True  → +1.0 for all agents
              mip_feasible=False → -2.0 for implicated agents, 0.0 for others
          - Store as an experience tuple in the replay buffer.
          - Optionally trigger a mini-batch gradient update.
        """
        self._stats["update_reward_calls"] = int(self._stats["update_reward_calls"]) + 1
        raise NotImplementedError(
            "MARLCommunitySolver.update_reward() is not yet implemented (Phase 5)."
        )

    def save_model(self, path: str) -> None:
        """
        Persist all agent Q-networks / policies to `path`.

        TODO (Phase 5):
          - torch.save({"agents": [...], "config": {...}}, path)
        """
        raise NotImplementedError(
            "MARLCommunitySolver.save_model() is not yet implemented (Phase 5)."
        )

    def load_model(self, path: str) -> None:
        """
        Load previously trained agent weights from `path`.

        TODO (Phase 5):
          - checkpoint = torch.load(path)
          - Restore agent weights.
          - self._is_trained = True
        """
        raise NotImplementedError(
            "MARLCommunitySolver.load_model() is not yet implemented (Phase 5)."
        )
