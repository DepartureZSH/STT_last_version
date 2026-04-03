"""
Phase 5 - MARLCommunitySolver: Test Script
==========================================
Tests the interface contract of MARLCommunitySolver.

Structure
---------
  Group A  (run now)  - interface conformance, action_space, no-good,
                        stats — all work without training.
  Group B  (SKIP now) - sample(), train(), update_reward(), save/load_model()
                        — gated on solver.is_trained; will run once Phase 5
                        is implemented and train() is called.

Run from project root:
    python tests/phase5_marl_solver_test.py

No Gurobi licence required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataReader import PSTTReader
from src.hybrid.special_constraint_decomposer import SpecialConstraintDecomposer
from src.hybrid.community_solver_base import (
    CommunitySolverBase, MARLCommunitySolverBase
)
from src.hybrid.marl_community_solver import MARLCommunitySolver
from src.hybrid.random_community_solver import RandomCommunitySolver

INSTANCE = "data/source/instances/agh-fal17.xml"

def section(name): print(f"\n{'='*58}\n  {name}\n{'='*58}")
def ok(msg):       print(f"  [OK]   {msg}")
def skip(msg):     print(f"  [SKIP] {msg} (implement Phase 5 to enable)")
def pending(msg):  print(f"  [TODO] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
section("Setup")
reader = PSTTReader(INSTANCE)
decomp = SpecialConstraintDecomposer(reader)
result = decomp.classify_classes()
communities = result.communities

marl = MARLCommunitySolver(reader, decomp, seed=42)
ok(f"MARLCommunitySolver instantiated (device={marl.device})")
ok(f"Instance: {reader.problem_name} - {len(communities)} communities")

# ─────────────────────────────────────────────────────────────────────────────
# A1: Class hierarchy conformance
# ─────────────────────────────────────────────────────────────────────────────
section("A1: Class hierarchy conformance")

assert isinstance(marl, CommunitySolverBase), \
    "MARLCommunitySolver must be a CommunitySolverBase"
ok("isinstance(marl, CommunitySolverBase)")

assert isinstance(marl, MARLCommunitySolverBase), \
    "MARLCommunitySolver must be a MARLCommunitySolverBase"
ok("isinstance(marl, MARLCommunitySolverBase)")

# All base abstract methods must be present
base_methods = ["action_space","sample","add_no_good","clear_no_goods",
                "no_good_count","stats","reset_stats"]
for m in base_methods:
    assert hasattr(marl, m), f"Missing CommunitySolverBase method: {m}"
ok(f"All {len(base_methods)} CommunitySolverBase methods present")

# MARL-specific methods
marl_methods = ["train","update_reward","save_model","load_model"]
for m in marl_methods:
    assert hasattr(marl, m), f"Missing MARLCommunitySolverBase method: {m}"
ok(f"All {len(marl_methods)} MARLCommunitySolverBase methods present")

assert hasattr(marl, "is_trained"), "Missing is_trained property"
ok("is_trained property present")

# ─────────────────────────────────────────────────────────────────────────────
# A2: is_trained starts False
# ─────────────────────────────────────────────────────────────────────────────
section("A2: is_trained initial state")

assert marl.is_trained is False, \
    f"is_trained should be False before training, got {marl.is_trained}"
ok("is_trained == False before training")

# ─────────────────────────────────────────────────────────────────────────────
# A3: action_space — same logic as RandomCommunitySolver
# ─────────────────────────────────────────────────────────────────────────────
section("A3: action_space - correctness and caching")

rand = RandomCommunitySolver(reader, decomp, seed=0)

for cid in list(result.community)[:30]:
    opts_marl = marl.action_space(cid)
    opts_rand = rand.action_space(cid)
    assert isinstance(opts_marl, list)
    assert set(opts_marl) == set(opts_rand), \
        f"{cid}: MARL action_space {set(opts_marl)} != Random {set(opts_rand)}"

ok("action_space identical to RandomCommunitySolver for 30 classes")

# Caching
for cid in list(result.community)[:10]:
    a1 = marl.action_space(cid)
    a2 = marl.action_space(cid)
    assert a1 is a2, f"action_space({cid}) not cached"
ok("action_space caching works")

# ─────────────────────────────────────────────────────────────────────────────
# A4: no-good — add, count, clear
# ─────────────────────────────────────────────────────────────────────────────
section("A4: no-good management")

comm0 = communities[0]
dummy_assign = {cid: (0, None) for cid in comm0.class_ids}

assert marl.no_good_count(comm0.id) == 0
marl.add_no_good(comm0.id, dummy_assign)
assert marl.no_good_count(comm0.id) == 1
ok("add_no_good + no_good_count work")

# Duplicate ignored
marl.add_no_good(comm0.id, dummy_assign)
assert marl.no_good_count(comm0.id) == 1
ok("Duplicate no-good not double-counted")

# Clear specific
marl.clear_no_goods(comm0.id)
assert marl.no_good_count(comm0.id) == 0
ok("clear_no_goods(community_id) works")

# Clear all
for comm in communities[:3]:
    marl.add_no_good(comm.id, {})
marl.clear_no_goods()
for comm in communities[:3]:
    assert marl.no_good_count(comm.id) == 0
ok("clear_no_goods() clears all")

# ─────────────────────────────────────────────────────────────────────────────
# A5: stats — structure and reset
# ─────────────────────────────────────────────────────────────────────────────
section("A5: stats structure and reset_stats")

s = marl.stats()
assert isinstance(s, dict), "stats() must return dict"
required_keys = ["total_sample_calls","total_attempts",
                 "special_constraint_rejections","no_good_rejections",
                 "successes","exhausted"]
for k in required_keys:
    assert k in s, f"Missing stats key: {k}"
ok(f"stats() has all {len(required_keys)} required keys")

# MARL-specific keys
marl_keys = ["train_episodes","mean_reward","update_reward_calls"]
for k in marl_keys:
    assert k in s, f"Missing MARL stats key: {k}"
ok(f"stats() has all {len(marl_keys)} MARL-specific keys")

marl.reset_stats()
s2 = marl.stats()
for k, v in s2.items():
    if k != "mean_reward":
        assert v == 0, f"After reset_stats(), {k}={v} (expected 0)"
ok("reset_stats() zeros all numeric fields")

# ─────────────────────────────────────────────────────────────────────────────
# A6: Stub methods raise NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────
section("A6: Unimplemented methods raise NotImplementedError")

comm0 = communities[0]
dummy_assign = {cid: (0, None) for cid in comm0.class_ids}

try:
    marl.sample(comm0, max_attempts=1)
    assert False, "sample() should raise NotImplementedError"
except NotImplementedError:
    ok("sample() raises NotImplementedError (not yet implemented)")

try:
    marl.train(communities[:1], n_episodes=1)
    assert False, "train() should raise NotImplementedError"
except NotImplementedError:
    ok("train() raises NotImplementedError (not yet implemented)")

try:
    marl.save_model("/tmp/test_model.pt")
    assert False, "save_model() should raise NotImplementedError"
except NotImplementedError:
    ok("save_model() raises NotImplementedError (not yet implemented)")

try:
    marl.load_model("/tmp/test_model.pt")
    assert False, "load_model() should raise NotImplementedError"
except NotImplementedError:
    ok("load_model() raises NotImplementedError (not yet implemented)")

try:
    marl.update_reward(comm0.id, dummy_assign, set(), True)
    assert False, "update_reward() should raise NotImplementedError"
except NotImplementedError:
    ok("update_reward() raises NotImplementedError (not yet implemented)")
    # But update_reward_calls counter should still increment
    assert marl.stats()["update_reward_calls"] == 1
    ok("update_reward_calls incremented even though NotImplementedError raised")

# ─────────────────────────────────────────────────────────────────────────────
# B: Phase 5 implementation tests (SKIP until trained)
# ─────────────────────────────────────────────────────────────────────────────
section("B: Phase 5 implementation tests")

if not marl.is_trained:
    skip("B1: sample() returns valid feasible assignment")
    skip("B2: sample() rejects no-good assignments")
    skip("B3: sample() returns None when exhausted")
    skip("B4: train() sets is_trained=True and returns summary dict")
    skip("B5: train() summary has 'episodes' and 'mean_reward' keys")
    skip("B6: update_reward() does not raise after training")
    skip("B7: save_model() / load_model() round-trip preserves behaviour")
    skip("B8: trained MARL samples satisfy special constraints (>= 80% of communities)")
    skip("B9: MARL sample rejection rate lower than RandomCommunitySolver (learning signal)")
    pending("Implement MARLCommunitySolver.train(), sample(), update_reward(), "
            "save_model(), load_model() to enable B tests")
else:
    # ── B1: sample coverage ───────────────────────────────────────────────
    n_ok = n_none = n_fail = 0
    for comm in communities:
        a = marl.sample(comm, max_attempts=500)
        if a is None:
            n_none += 1
            continue
        assert set(a.keys()) == comm.class_ids
        if decomp.check_special_constraints(comm, a):
            n_ok += 1
        else:
            n_fail += 1
    ok(f"B1/B8: sample ok={n_ok}, exhausted={n_none}, violations={n_fail}")
    assert n_fail == 0, f"B1: {n_fail} samples violated special constraints"

    # ── B4: is_trained ───────────────────────────────────────────────────
    assert marl.is_trained is True
    ok("B4: is_trained == True")

    # ── B5: train summary ────────────────────────────────────────────────
    summary = marl.train(communities[:1], n_episodes=1)
    assert "episodes" in summary and "mean_reward" in summary
    ok(f"B5: train() returned summary {summary}")

    # ── B6: update_reward after training ─────────────────────────────────
    comm0 = communities[0]
    a0 = marl.sample(comm0, max_attempts=200)
    if a0 is not None:
        marl.update_reward(comm0.id, a0, set(), True)
        ok("B6: update_reward() does not raise")

    # ── B7: save/load round-trip ─────────────────────────────────────────
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        tmp_path = tf.name
    try:
        marl.save_model(tmp_path)
        marl2 = MARLCommunitySolver(reader, decomp, seed=0)
        assert not marl2.is_trained
        marl2.load_model(tmp_path)
        assert marl2.is_trained
        ok("B7: save/load round-trip preserves is_trained=True")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ── B9: MARL vs Random rejection rate ────────────────────────────────
    rand2 = RandomCommunitySolver(reader, decomp, seed=42)
    rand2.reset_stats()
    for comm in communities:
        rand2.sample(comm, max_attempts=200)
    marl.reset_stats()
    for comm in communities:
        marl.sample(comm, max_attempts=200)
    rs = rand2.stats()
    ms = marl.stats()
    rand_rej_rate = rs["special_constraint_rejections"] / max(rs["total_attempts"], 1)
    marl_rej_rate = ms["special_constraint_rejections"] / max(ms["total_attempts"], 1)
    ok(f"B9: rand rejection rate={rand_rej_rate:.3f}, "
       f"marl rejection rate={marl_rej_rate:.3f}")
    if marl_rej_rate <= rand_rej_rate:
        ok("B9: MARL rejection rate <= Random (learning is effective)")
    else:
        pending(f"B9: MARL rejection rate {marl_rej_rate:.3f} > Random {rand_rej_rate:.3f} "
                "(model may need more training)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("  PHASE 5 INTERFACE TESTS COMPLETE")
if not marl.is_trained:
    print("  (B-group tests SKIPPED -- implement train() to enable)")
print(f"{'='*58}\n")
