"""
Phase 2 - RandomCommunitySolver: Test Script
=============================================
Run from project root:
    python tests/phase2_random_solver_test.py

No Gurobi licence required.
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataReader import PSTTReader
from src.hybrid.special_constraint_decomposer import SpecialConstraintDecomposer, _room_available
from src.hybrid.random_community_solver import (
    RandomCommunitySolver, _assignment_to_nogood
)
from src.hybrid.community_solver_base import CommunitySolverBase

# ── Instance with special constraints ────────────────────────────────────────
INSTANCE = "data/source/instances/agh-fal17.xml"

def section(name): print(f"\n{'='*58}\n  {name}\n{'='*58}")
def ok(msg):   print(f"  [OK] {msg}")
def skip(msg): print(f"  [--] SKIP: {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
section("Setup")
reader = PSTTReader(INSTANCE)
decomp = SpecialConstraintDecomposer(reader)
result = decomp.classify_classes()
communities = result.communities
solver = RandomCommunitySolver(reader, decomp, seed=42)
ok(f"Instance: {reader.problem_name} - {len(reader.classes)} classes, "
   f"{len(communities)} communities, {len(result.community)} community classes")

# ─────────────────────────────────────────────────────────────────────────────
# T1: CommunitySolverBase interface conformance
# ─────────────────────────────────────────────────────────────────────────────
section("T1: CommunitySolverBase interface conformance")

assert isinstance(solver, CommunitySolverBase), \
    "RandomCommunitySolver must be an instance of CommunitySolverBase"
ok("isinstance(solver, CommunitySolverBase) OK")

for method in ["action_space","sample","add_no_good","clear_no_goods",
               "no_good_count","stats","reset_stats"]:
    assert hasattr(solver, method), f"Missing method: {method}"
ok("All 7 abstract methods present OK")

# ─────────────────────────────────────────────────────────────────────────────
# T2: action_space - validity of returned (tidx, rid) pairs
# ─────────────────────────────────────────────────────────────────────────────
section("T2: action_space - option validity")

n_checked = 0
for cid in list(result.community)[:50]:   # sample 50 community classes
    cls  = reader.classes[cid]
    opts = solver.action_space(cid)

    assert isinstance(opts, list), f"action_space({cid}) must return list"

    room_required = cls.get("room_required", True)
    room_opts     = {r["id"] for r in cls.get("room_options", [])}
    n_time_opts   = len(cls["time_options"])

    for tidx, rid in opts:
        # tidx in range
        assert 0 <= tidx < n_time_opts, \
            f"{cid}: tidx={tidx} out of range [0,{n_time_opts})"
        # rid validity
        if room_required and room_opts:
            assert rid in room_opts, \
                f"{cid}: rid={rid} not in room_options"
            # room must be available at this time
            tb = cls["time_options"][tidx]["optional_time_bits"]
            room_data = reader.rooms.get(rid, {})
            assert _room_available(room_data, tb), \
                f"{cid}: room {rid} unavailable at time {tb}"
        else:
            assert rid is None, f"{cid}: room_not_required but rid={rid}"
    n_checked += 1

ok(f"action_space valid options for {n_checked} classes OK")

# ─────────────────────────────────────────────────────────────────────────────
# T3: action_space - caching
# ─────────────────────────────────────────────────────────────────────────────
section("T3: action_space - caching")

for cid in list(result.community)[:20]:
    a1 = solver.action_space(cid)
    a2 = solver.action_space(cid)
    assert a1 is a2, f"action_space({cid}) not cached (different object)"
ok("action_space returns same object on repeat calls OK")

# ─────────────────────────────────────────────────────────────────────────────
# T4: action_space - no-room classes get rid=None
# ─────────────────────────────────────────────────────────────────────────────
section("T4: action_space - room-not-required classes")

no_room_cids = [cid for cid in reader.classes
                if not reader.classes[cid].get("room_required", True)
                or not reader.classes[cid].get("room_options")]
if no_room_cids:
    for cid in no_room_cids[:10]:
        opts = solver.action_space(cid)
        for _, rid in opts:
            assert rid is None, f"{cid}: expected rid=None, got {rid}"
    ok(f"rid=None for {len(no_room_cids)} room-not-required classes OK")
else:
    skip("No room-not-required classes in this instance")

# ─────────────────────────────────────────────────────────────────────────────
# T5: sample - covers all community class IDs
# ─────────────────────────────────────────────────────────────────────────────
section("T5: sample - coverage")

for comm in communities:
    a = solver.sample(comm, max_attempts=500)
    if a is None:
        continue   # exhausted - tested separately
    assert set(a.keys()) == comm.class_ids, \
        f"Community {comm.id}: assignment covers {set(a.keys())} ≠ {comm.class_ids}"
ok(f"Assignments cover all class_ids for all {len(communities)} communities OK")

# ─────────────────────────────────────────────────────────────────────────────
# T6: sample - returned (tidx, rid) pairs are in action_space
# ─────────────────────────────────────────────────────────────────────────────
section("T6: sample - options within action_space")

solver6 = RandomCommunitySolver(reader, decomp, seed=0)
n_violations = 0
for comm in communities:
    a = solver6.sample(comm, max_attempts=500)
    if a is None:
        continue
    for cid, (tidx, rid) in a.items():
        valid = solver6.action_space(cid)
        if (tidx, rid) not in valid:
            n_violations += 1

assert n_violations == 0, f"{n_violations} assignments outside action_space"
ok(f"All sampled (tidx, rid) pairs within action_space OK")

# ─────────────────────────────────────────────────────────────────────────────
# T7: sample - satisfies special constraints
# ─────────────────────────────────────────────────────────────────────────────
section("T7: sample - special constraint satisfaction")

solver7 = RandomCommunitySolver(reader, decomp, seed=1)
n_pass = n_fail = n_none = 0
for comm in communities:
    a = solver7.sample(comm, max_attempts=1000)
    if a is None:
        n_none += 1
        continue
    ok_val = decomp.check_special_constraints(comm, a)
    if ok_val:
        n_pass += 1
    else:
        n_fail += 1

assert n_fail == 0, f"{n_fail} samples violated special constraints"
ok(f"Constraint check: pass={n_pass}, exhausted={n_none}, violations={n_fail} OK")

# ─────────────────────────────────────────────────────────────────────────────
# T8: sample - seed reproducibility
# ─────────────────────────────────────────────────────────────────────────────
section("T8: sample - seed reproducibility")

comm0 = communities[0]
s_a = RandomCommunitySolver(reader, decomp, seed=99)
s_b = RandomCommunitySolver(reader, decomp, seed=99)
a_a = s_a.sample(comm0, max_attempts=500)
a_b = s_b.sample(comm0, max_attempts=500)
assert a_a == a_b, "Same seed must produce same sample"
ok(f"Same seed -> same sample for community {comm0.id} OK")

s_c = RandomCommunitySolver(reader, decomp, seed=100)
a_c = s_c.sample(comm0, max_attempts=500)
if a_a is not None and a_c is not None:
    # Different seeds should (very likely) produce different samples
    # (not guaranteed, but almost certain with >1 valid assignment)
    if a_a != a_c:
        ok("Different seeds -> different samples OK")
    else:
        skip("Different seeds produced the same sample (possible, not an error)")

# ─────────────────────────────────────────────────────────────────────────────
# T9: sample - constraint-ordered (most constrained first)
# ─────────────────────────────────────────────────────────────────────────────
section("T9: _order_by_constraint - ascending action-space size")

for comm in communities[:5]:
    ordered = solver._order_by_constraint(comm)
    sizes   = [len(solver.action_space(cid)) for cid in ordered]
    assert sizes == sorted(sizes), \
        f"Community {comm.id}: ordering not ascending: {sizes}"
ok("Classes ordered by ascending action-space size OK")

# ─────────────────────────────────────────────────────────────────────────────
# T10: no-good - add and reject
# ─────────────────────────────────────────────────────────────────────────────
section("T10: no-good - add_no_good + rejection")

solver10 = RandomCommunitySolver(reader, decomp, seed=7)
comm0 = communities[0]

# Get a valid assignment
a = solver10.sample(comm0, max_attempts=500)
if a is None:
    skip("Community 0 exhausted - T10 skipped")
else:
    assert solver10.no_good_count(comm0.id) == 0

    solver10.add_no_good(comm0.id, a)
    assert solver10.no_good_count(comm0.id) == 1
    ok("add_no_good increments count OK")

    # The no-good key must appear in the set
    ng = _assignment_to_nogood(a)
    assert ng in solver10._no_goods[comm0.id]
    ok("Assignment key present in _no_goods OK")

    # New samples should not return the no-good assignment
    # (sample enough times to be statistically confident)
    for _ in range(200):
        a2 = solver10.sample(comm0, max_attempts=500)
        if a2 is not None:
            assert a2 != a, "No-good assignment was returned by sample()"
    ok("No-good assignment not returned in 200 subsequent samples OK")

# ─────────────────────────────────────────────────────────────────────────────
# T11: no-good - multiple entries, count
# ─────────────────────────────────────────────────────────────────────────────
section("T11: no-good - multiple entries")

solver11 = RandomCommunitySolver(reader, decomp, seed=8)
comm0 = communities[0]
seen = set()
for _ in range(5):
    a = solver11.sample(comm0, max_attempts=500)
    if a is None:
        break
    solver11.add_no_good(comm0.id, a)
    seen.add(_assignment_to_nogood(a))

assert solver11.no_good_count(comm0.id) == len(seen)
ok(f"no_good_count = {len(seen)} unique no-goods OK")

# Adding duplicate no-good does not double-count
if seen:
    first_ng_a = {cid: (tidx, rid) for cid, tidx, rid in next(iter(seen))}
    solver11.add_no_good(comm0.id, first_ng_a)
    assert solver11.no_good_count(comm0.id) == len(seen)
    ok("Duplicate no-good not double-counted OK")

# ─────────────────────────────────────────────────────────────────────────────
# T12: clear_no_goods
# ─────────────────────────────────────────────────────────────────────────────
section("T12: clear_no_goods")

solver12 = RandomCommunitySolver(reader, decomp, seed=9)
for comm in communities[:3]:
    a = solver12.sample(comm, max_attempts=300)
    if a:
        solver12.add_no_good(comm.id, a)

# Clear one community
comm_id = communities[0].id
solver12.clear_no_goods(comm_id)
assert solver12.no_good_count(comm_id) == 0
ok(f"clear_no_goods(community_id={comm_id}) clears only that community OK")
# Others still intact
for comm in communities[1:3]:
    if solver12.no_good_count(comm.id) > 0:
        ok(f"Community {comm.id} no-goods intact OK")

# Clear all
solver12.clear_no_goods()
for comm in communities[:3]:
    assert solver12.no_good_count(comm.id) == 0
ok("clear_no_goods() clears all communities OK")

# ─────────────────────────────────────────────────────────────────────────────
# T13: stats - tracking
# ─────────────────────────────────────────────────────────────────────────────
section("T13: stats - correct tracking")

solver13 = RandomCommunitySolver(reader, decomp, seed=13)
solver13.reset_stats()

total_calls = 0
for comm in communities[:5]:
    solver13.sample(comm, max_attempts=200)
    total_calls += 1

s = solver13.stats()
assert s["total_sample_calls"] == total_calls
assert s["total_attempts"] >= total_calls   # at least one attempt per call
assert s["total_attempts"] == (
    s["special_constraint_rejections"] +
    s["no_good_rejections"] +
    s["successes"]
), "attempts must equal sum of outcomes"
ok(f"stats: calls={s['total_sample_calls']}, attempts={s['total_attempts']}, "
   f"sc_rej={s['special_constraint_rejections']}, ng_rej={s['no_good_rejections']}, "
   f"ok={s['successes']}, exhausted={s['exhausted']} OK")

# ─────────────────────────────────────────────────────────────────────────────
# T14: reset_stats
# ─────────────────────────────────────────────────────────────────────────────
section("T14: reset_stats")

solver13.reset_stats()
s = solver13.stats()
for k, v in s.items():
    assert v == 0, f"After reset_stats(), {k}={v} (expected 0)"
ok("All stats zeroed after reset_stats() OK")

# ─────────────────────────────────────────────────────────────────────────────
# T15: sample returns None when max_attempts=0
# ─────────────────────────────────────────────────────────────────────────────
section("T15: sample returns None when max_attempts=0")

solver15 = RandomCommunitySolver(reader, decomp, seed=0)
comm0 = communities[0]
result15 = solver15.sample(comm0, max_attempts=0)
assert result15 is None, "max_attempts=0 must return None"
ok("sample(max_attempts=0) returns None OK")

# ─────────────────────────────────────────────────────────────────────────────
# T16: cross-instance - all instances with communities
# ─────────────────────────────────────────────────────────────────────────────
section("T16: cross-instance sampling check")

paths = sorted(glob.glob("data/source/**/*.xml", recursive=True))
n_instances_tested = 0
for path in paths:
    r = PSTTReader(path)
    d = SpecialConstraintDecomposer(r)
    res = d.classify_classes()
    if not res.communities:
        continue
    s = RandomCommunitySolver(r, d, seed=0)
    for comm in res.communities:
        a = s.sample(comm, max_attempts=300)
        if a is not None:
            assert set(a.keys()) == comm.class_ids
            assert d.check_special_constraints(comm, a)
    n_instances_tested += 1

ok(f"Cross-instance: {n_instances_tested} instances with communities all passed OK")

# ─────────────────────────────────────────────────────────────────────────────
# T17: _community_action_space_size
# ─────────────────────────────────────────────────────────────────────────────
section("T17: _community_action_space_size")

for comm in communities[:5]:
    size = solver._community_action_space_size(comm)
    assert isinstance(size, int) and size >= 0
    # Manual check: product of individual sizes
    expected = 1
    for cid in comm.class_ids:
        n = len(solver.action_space(cid))
        if n == 0:
            expected = 0
            break
        expected *= n
    assert size == expected, f"Community {comm.id}: size={size} != {expected}"
ok("_community_action_space_size matches product of individual sizes OK")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("  ALL PHASE 2 TESTS PASSED")
print(f"{'='*58}\n")
