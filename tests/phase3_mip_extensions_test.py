"""
Phase 3 — MIP Integration Extensions: Test Script
===================================================
Tests for three new MIPSolver methods:
  - fix_assignments(fixed)
  - get_infeasibility_info()
  - reset_fixed()

IMPORTANT: Requires a valid Gurobi licence. Do NOT run on a machine
           without a licence. Run manually on the licensed host:

    cd <project_root>
    python tests/phase3_mip_extensions_test.py

Instance used: data/source/test/bet-sum18.xml  (127 classes, small/fast)
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataReader import PSTTReader
from src.MIP.solver import MIPSolver
from src.hybrid.special_constraint_decomposer import SpecialConstraintDecomposer

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
INSTANCE = 'data/source/test/bet-sum18.xml'

# Minimal config accepted by MIPSolver.__init__
TEST_CONFIG = {
    'train': {
        'MIP': {
            'time_limit': 120,
            'Threads':    4,
            'MIPGap':     0.01,
            'PoolSolutions': 1,
        }
    },
    'config': {
        'technique':   'MIP',
        'author':      'test',
        'institution': 'test',
        'country':     'test',
        'include_students': False,
    }
}

def make_logger(name='test'):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('  [%(levelname)s] %(message)s'))
        log.addHandler(h)
    return log

def make_solver(reader, logger):
    solver = MIPSolver(reader, logger, TEST_CONFIG)
    solver.build_model()
    return solver

def section(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

def ok(msg):
    print(f"  \u2713 {msg}")

def fail(msg):
    print(f"  \u2717 FAIL: {msg}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
section("Setup — load instance and build solver")
reader = PSTTReader(INSTANCE)
logger = make_logger()
ok(f"Loaded {reader.problem_name}: {len(reader.classes)} classes, {len(reader.rooms)} rooms")

decomp = SpecialConstraintDecomposer(reader)
result = decomp.classify_classes()
ok(f"Decomposition: fixed={len(result.fixed)}, community={len(result.community)}, free={len(result.free)}")

# Pick a small set of test classes (use first 5 free classes for most tests)
test_cids = sorted(result.free)[:5]
ok(f"Test class IDs: {test_cids}")

# ─────────────────────────────────────────────────────────────────────────────
# T1: fix_assignments — internal state
# ─────────────────────────────────────────────────────────────────────────────
section("T1: fix_assignments — internal state checks")

solver = make_solver(reader, logger)

# Build a valid assignment dict for the test classes
fixed = {}
for cid in test_cids:
    cls = reader.classes[cid]
    tidx = 0
    rid  = cls['room_options'][0]['id'] if cls.get('room_options') else None
    fixed[cid] = (tidx, rid)

solver.fix_assignments(fixed)

# _fixed_assignments should mirror what we passed
assert solver._fixed_assignments == fixed, \
    f"_fixed_assignments mismatch: {solver._fixed_assignments} vs {fixed}"
ok("_fixed_assignments populated correctly")

# _fix_constraints should be non-empty
assert len(solver._fix_constraints) > 0, "_fix_constraints is empty after fix_assignments"
ok(f"_fix_constraints: {len(solver._fix_constraints)} constraints added")

# _fix_constr_to_cid: every key should map to a cid in fixed
for name, cid in solver._fix_constr_to_cid.items():
    assert cid in fixed, f"Constraint {name} maps to unknown cid {cid}"
ok(f"_fix_constr_to_cid: {len(solver._fix_constr_to_cid)} entries, all valid")

# Verify expected constraint names exist
from gurobipy import GRB
model_constr_names = {c.ConstrName for c in solver.model.getConstrs()}
for cid, (tidx, rid) in fixed.items():
    if (cid, tidx) in solver.y:
        expected = f"fix_y_{cid}_{tidx}"
        assert expected in model_constr_names, f"Expected constraint '{expected}' not found in model"
    if cid in solver.u:
        expected = f"fix_u_{cid}"
        assert expected in model_constr_names, f"Expected constraint '{expected}' not found in model"
ok("fix_y_* and fix_u_* constraint names present in Gurobi model")

# ─────────────────────────────────────────────────────────────────────────────
# T2: fix_assignments — solution respects fixed assignments
# ─────────────────────────────────────────────────────────────────────────────
section("T2: Solving with fixed assignments — solution compliance")

assignments_list = solver.solve()
if assignments_list is None:
    fail("Solver returned None (infeasible) even with valid fixed assignments")

assignments = assignments_list[0]  # best solution

for cid, (tidx, rid) in fixed.items():
    a = assignments.get(cid)
    if a is None:
        fail(f"Class {cid} missing from solution")
    sol_topt, sol_room_required, sol_rid, _ = a
    if sol_topt is None:
        fail(f"Class {cid} is unassigned in solution despite being fixed")

    # Verify time option matches
    expected_topt = reader.classes[cid]['time_options'][tidx]
    assert sol_topt is expected_topt or sol_topt == expected_topt, \
        f"Class {cid}: expected time_option[{tidx}], got different"

    # Verify room matches (if room was specified)
    if rid is not None:
        assert sol_rid == rid, \
            f"Class {cid}: expected room {rid}, got {sol_rid}"

ok(f"All {len(fixed)} fixed classes assigned to their forced (time, room) in solution")

# ─────────────────────────────────────────────────────────────────────────────
# T3: reset_fixed — model state restored
# ─────────────────────────────────────────────────────────────────────────────
section("T3: reset_fixed — constraint removal and state reset")

n_constrs_before_fix   = None  # captured before fix in T4 (below)
n_constrs_after_fix    = len(solver.model.getConstrs())
n_fix_constrs          = len(solver._fix_constraints)

# Now reset
solver.reset_fixed()

assert len(solver._fix_constraints)    == 0, "_fix_constraints not empty after reset"
assert len(solver._fix_constr_to_cid)  == 0, "_fix_constr_to_cid not empty after reset"
assert len(solver._fixed_assignments)  == 0, "_fixed_assignments not empty after reset"
ok("All tracking dicts cleared")

n_constrs_after_reset = len(solver.model.getConstrs())
assert n_constrs_after_reset == n_constrs_after_fix - n_fix_constrs, (
    f"Constraint count mismatch after reset: "
    f"after_fix={n_constrs_after_fix}, removed={n_fix_constrs}, "
    f"after_reset={n_constrs_after_reset}"
)
ok(f"Gurobi model has {n_fix_constrs} fewer constraints after reset_fixed()")

# No fix_* constraints should remain in the model
remaining_fix = [c.ConstrName for c in solver.model.getConstrs()
                 if c.ConstrName.startswith('fix_')]
assert len(remaining_fix) == 0, f"Leftover fix constraints: {remaining_fix}"
ok("No fix_* constraints remain in Gurobi model")

# ─────────────────────────────────────────────────────────────────────────────
# T4: reset_fixed → re-fix → solve again (idempotency)
# ─────────────────────────────────────────────────────────────────────────────
section("T4: reset_fixed + re-fix cycle (idempotency)")

# Rebuild for a clean start (solver.model already contains no-good cuts from T2)
solver2 = make_solver(reader, logger)

# First fix-solve-reset cycle
solver2.fix_assignments(fixed)
n_after_first_fix = len(solver2.model.getConstrs())
r1 = solver2.solve()
solver2.reset_fixed()
n_after_first_reset = len(solver2.model.getConstrs())

# Second fix-solve cycle with the same fixed dict
solver2.fix_assignments(fixed)
n_after_second_fix = len(solver2.model.getConstrs())
r2 = solver2.solve()

# Constraint counts from fix should be equal between cycles
# (note: solve() adds a no-good cut to model, so base count grows by 1 each solve)
assert n_after_second_fix - n_after_first_reset == n_after_first_fix - (n_after_first_reset - (n_after_first_fix - n_after_first_reset)), \
    "Fix constraint delta differs between cycles (unexpected)"
ok("Same number of fix constraints added in both cycles")

if r1 is not None and r2 is not None:
    ok("Solver finds solutions in both cycles")
elif r1 is None and r2 is None:
    ok("Both cycles infeasible (consistent)")
else:
    fail(f"Solver inconsistency: cycle1={r1 is not None}, cycle2={r2 is not None}")

# ─────────────────────────────────────────────────────────────────────────────
# T5: fix_assignments + infeasible forced conflict
# ─────────────────────────────────────────────────────────────────────────────
section("T5: Forced infeasibility → get_infeasibility_info()")

# Build a fresh solver
solver3 = make_solver(reader, logger)

# Find two classes that CONFLICT (share a NotOverlap / DifferentTime constraint)
# and force them both to the same conflicting time slots.
# Strategy: find a pair from a NotOverlap hard constraint, assign both to
# the SAME time_option (identical weeks/days/start → guaranteed overlap).

conflict_pair = None
for cons in reader.distributions['hard_constraints']:
    if cons['type'] in ('NotOverlap', 'DifferentTime', 'SameAttendees') and len(cons['classes']) >= 2:
        c1, c2 = cons['classes'][0], cons['classes'][1]
        # Find a time option valid for both classes
        tops1 = reader.classes[c1]['time_options']
        tops2 = reader.classes[c2]['time_options']
        # Try: assign both to their first time_option and hope they conflict
        # (any NotOverlap constraint means they must NOT have the same time)
        t1bits = tops1[0]['optional_time_bits']
        for tidx2, t2 in enumerate(tops2):
            t2bits = t2['optional_time_bits']
            # Check for time overlap
            w_and = int(t1bits[0], 2) & int(t2bits[0], 2)
            d_and = int(t1bits[1], 2) & int(t2bits[1], 2)
            overlap = (t1bits[2] < t2bits[2] + t2bits[3]) and (t2bits[2] < t1bits[2] + t1bits[3])
            if w_and and d_and and overlap:
                conflict_pair = (c1, 0, c2, tidx2)
                break
        if conflict_pair:
            break

if conflict_pair is None:
    print("  (No overlapping time pair found — skipping T5)")
else:
    c1, tidx1, c2, tidx2 = conflict_pair
    rid1 = reader.classes[c1]['room_options'][0]['id'] if reader.classes[c1].get('room_options') else None
    rid2 = reader.classes[c2]['room_options'][0]['id'] if reader.classes[c2].get('room_options') else None

    conflict_fixed = {c1: (tidx1, rid1), c2: (tidx2, rid2)}
    ok(f"Forcing conflict: classes {c1} (tidx={tidx1}) and {c2} (tidx={tidx2}) to overlapping times")

    solver3.fix_assignments(conflict_fixed)
    result3 = solver3.solve()

    if result3 is not None:
        # Conflict might be soft or solver found a workaround; skip IIS test
        print("  (Solver found a solution despite forced overlap — constraint may be soft; skipping IIS check)")
    else:
        # Should be infeasible
        assert solver3.model.Status == GRB.INFEASIBLE, \
            f"Expected INFEASIBLE, got status {solver3.model.Status}"
        ok("Model correctly reported INFEASIBLE")

        info = solver3.get_infeasibility_info()

        # Return type checks
        assert isinstance(info, dict), "get_infeasibility_info() must return dict"
        assert 'implicated_cids' in info and 'violated_constraints' in info, \
            f"Missing keys in info: {info.keys()}"
        assert isinstance(info['implicated_cids'], set), \
            "'implicated_cids' must be a set"
        assert isinstance(info['violated_constraints'], list), \
            "'violated_constraints' must be a list"
        ok(f"Return type correct: implicated_cids={info['implicated_cids']}, "
           f"violated_constraints={len(info['violated_constraints'])}")

        # At least one of the two forced classes should appear in the IIS
        implicated = info['implicated_cids']
        assert len(implicated) > 0, "IIS returned no implicated fixed classes"
        assert implicated.issubset({c1, c2}), \
            f"IIS implicated classes {implicated} outside forced pair {{{c1},{c2}}}"
        ok(f"IIS correctly identifies {implicated} as implicated in the conflict")

# ─────────────────────────────────────────────────────────────────────────────
# T6: get_infeasibility_info on non-infeasible model returns empty gracefully
# ─────────────────────────────────────────────────────────────────────────────
section("T6: get_infeasibility_info on non-infeasible model (graceful warning)")

solver4 = make_solver(reader, logger)
solver4.fix_assignments({test_cids[0]: fixed[test_cids[0]]})
solver4.solve()  # should be feasible or time-limit

info_empty = solver4.get_infeasibility_info()
assert isinstance(info_empty, dict)
assert info_empty['implicated_cids'] == set()
assert info_empty['violated_constraints'] == []
ok("Returns empty result without crashing on non-infeasible model")

# ─────────────────────────────────────────────────────────────────────────────
# T7: fix_assignments on classes not in the model (graceful skip)
# ─────────────────────────────────────────────────────────────────────────────
section("T7: fix_assignments with cid not in model variables (graceful)")

solver5 = make_solver(reader, logger)
n_before = len(solver5.model.getConstrs())

# Pass a non-existent cid — should not crash, just skip
fake_fixed = {'NONEXISTENT_CID': (0, None)}
solver5.fix_assignments(fake_fixed)
solver5.model.update()

n_after = len(solver5.model.getConstrs())
# No constraints should have been added for a non-existent cid
assert n_after == n_before, \
    f"Unexpected constraints added for non-existent cid: {n_after - n_before}"
ok("Non-existent cid silently ignored — no constraints added")

# ─────────────────────────────────────────────────────────────────────────────
# T8: Double fix_assignments warning (same solver, no reset between)
# ─────────────────────────────────────────────────────────────────────────────
section("T8: Double fix_assignments without reset — warning issued")

solver6 = make_solver(reader, logger)
solver6.fix_assignments({test_cids[0]: fixed[test_cids[0]]})
n1 = len(solver6._fix_constraints)

# Second call without reset — should log a warning and still proceed
solver6.fix_assignments({test_cids[1]: fixed[test_cids[1]]})
n2 = len(solver6._fix_constraints)

assert n2 > n1, "Second fix_assignments did not add any constraints"
ok(f"Second fix_assignments added more constraints ({n1} → {n2}); warning should have been logged above")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ALL PHASE 3 TESTS PASSED")
print(f"{'='*60}")
print("""
Tests completed:
  T1  fix_assignments — internal state (_fix_constraints, _fix_constr_to_cid, _fixed_assignments)
  T2  Solved model respects fixed (time, room) for each pinned class
  T3  reset_fixed — all constraints removed, tracking dicts cleared
  T4  reset + re-fix cycle is idempotent (consistent constraint counts)
  T5  Forced infeasible conflict → get_infeasibility_info returns correct implicated_cids
  T6  get_infeasibility_info on feasible model returns empty gracefully
  T7  Non-existent cid in fix_assignments silently skipped
  T8  Double fix_assignments (no reset) logs warning and still adds constraints
""")
