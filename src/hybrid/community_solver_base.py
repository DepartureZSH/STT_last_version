"""
community_solver_base.py
========================
Abstract base class (interface contract) shared by all community solvers:
  - RandomCommunitySolver  (Phase 2 — implemented)
  - MARLCommunitySolver    (Phase 5 — planned)

Any future solver that plugs into HybridSolver must subclass
CommunitySolverBase and implement every abstract method.

MARL-specific lifecycle methods (train / update_reward / save_model /
load_model) are defined in the MARLCommunitySolverBase sub-interface so
that the pure-random solver does not need to carry them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from src.hybrid.special_constraint_decomposer import Community

# {cid: (tidx, rid_or_None)}
Assignments = Dict[str, Tuple[int, Optional[str]]]


# ─────────────────────────────────────────────────────────────────────────────
# Core interface — all community solvers
# ─────────────────────────────────────────────────────────────────────────────
class CommunitySolverBase(ABC):
    """
    Interface contract for Layer-1 community assignment solvers.

    Concrete implementations must supply:
      action_space  — enumerate valid (tidx, rid) pairs per class
      sample        — draw a feasible, non-no-good community assignment
      add_no_good   — register a forbidden assignment (from MIP feedback)
      clear_no_goods — remove no-good cuts
      no_good_count — query number of registered no-goods
      stats         — return sampling statistics
      reset_stats   — zero out statistics
    """

    @abstractmethod
    def action_space(self, cid: str) -> List[Tuple[int, Optional[str]]]:
        """
        Return all valid (tidx, rid) pairs for class `cid`.
        rid is None for room-not-required classes.
        """

    @abstractmethod
    def sample(
        self,
        community: Community,
        max_attempts: int = 1000,
    ) -> Optional[Assignments]:
        """
        Draw a random / learned assignment for all classes in `community`.

        Returns
        -------
        Assignments  — {cid: (tidx, rid)} satisfying all special constraints
                       and not in the no-good set.
        None         — if no valid assignment can be found within max_attempts.
        """

    @abstractmethod
    def add_no_good(self, community_id: int, assignment: Assignments) -> None:
        """Register `assignment` as forbidden for community `community_id`."""

    @abstractmethod
    def clear_no_goods(self, community_id: Optional[int] = None) -> None:
        """
        Clear no-good cuts.
        If community_id is None, clear all; otherwise clear only that community.
        """

    @abstractmethod
    def no_good_count(self, community_id: int) -> int:
        """Return the number of no-good cuts for community `community_id`."""

    @abstractmethod
    def stats(self) -> dict:
        """Return a copy of sampling/training statistics."""

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset all statistics to zero."""


# ─────────────────────────────────────────────────────────────────────────────
# MARL-specific extension interface  (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
class MARLCommunitySolverBase(CommunitySolverBase):
    """
    Extended interface for MARL-based community solvers.

    Adds lifecycle methods for training, reward feedback, and model
    persistence on top of the core CommunitySolverBase contract.

    Reward signal design (guideline for implementors):
      +1.0  per special constraint satisfied by a community assignment
      -1.0  per special constraint violated
      -2.0  additional penalty for each agent (class) implicated in the MIP IIS
    """

    @abstractmethod
    def train(
        self,
        communities: list,
        n_episodes: int,
        *,
        verbose: bool = False,
    ) -> dict:
        """
        Run the MARL training loop against the special-constraint environment.

        Parameters
        ----------
        communities : list[Community]
        n_episodes  : number of training episodes
        verbose     : print per-episode statistics

        Returns
        -------
        Training summary dict: {"episodes": int, "mean_reward": float, ...}
        """

    @abstractmethod
    def update_reward(
        self,
        community_id:    int,
        assignment:      Assignments,
        implicated_cids: Set[str],
        mip_feasible:    bool,
    ) -> None:
        """
        Deliver a MIP-feedback reward signal to the agents in community
        `community_id`.

        Called by HybridSolver after each MIP solve:
          - mip_feasible=True  → positive reward for non-implicated agents
          - mip_feasible=False → negative reward for implicated_cids

        Parameters
        ----------
        community_id    : Community.id
        assignment      : the assignment that was tried
        implicated_cids : set of class IDs identified in the IIS
        mip_feasible    : whether the MIP found a solution
        """

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Persist trained agent weights/policies to `path`."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load previously trained agent weights/policies from `path`."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True once the model has been trained or loaded."""
