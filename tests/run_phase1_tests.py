"""
Standalone script that executes all logic tested by phase1_decomposer_test.ipynb.
Run from the project root:  python tests/run_phase1_tests.py
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataReader import PSTTReader
from src.hybrid.special_constraint_decomposer import (
    SpecialConstraintDecomposer, _parse_constraint_type, _is_special,
    _merge_slots, _UnionFind,
    _check_max_days, _check_max_day_load, _check_max_breaks, _check_max_block,
)

def tb(day_idx, start, length, nw=6):
    weeks = '1' * nw
    days  = '0'*day_idx + '1' + '0'*(6-day_idx)
    return (weeks, days, start, length)

MON, TUE, WED, FRI = 0, 1, 2, 4
NW, ND = 6, 7

def section(name): print(f"\n{'='*50}\n  {name}\n{'='*50}")

# ─────────────────────────────────────────────────
# Section 1: Helper functions
# ─────────────────────────────────────────────────
section("1. Helper functions")

for ts, exp in [
    ('MaxDays(3)',      ('MaxDays',    '3')),
    ('MaxDayLoad(84)',  ('MaxDayLoad', '84')),
    ('MaxBreaks(1,6)',  ('MaxBreaks',  '1,6')),
    ('MaxBlock(24,6)',  ('MaxBlock',   '24,6')),
    ('NotOverlap',      ('NotOverlap', None)),
    ('WorkDay(24)',     ('WorkDay',    '24')),
]:
    assert _parse_constraint_type(ts) == exp, f"Failed: {ts}"
print("  _parse_constraint_type  OK")

for t in ['MaxDays(3)','MaxDayLoad(84)','MaxBreaks(1,6)','MaxBlock(24,6)']:
    assert _is_special(t), f"Should be special: {t}"
for t in ['NotOverlap','SameTime','WorkDay(24)','MinGap(6)']:
    assert not _is_special(t), f"Should NOT be special: {t}"
print("  _is_special             OK")

for slots, gap, exp in [
    ([(0,2),(3,2)],       1, [(0,5)]),
    ([(0,2),(4,2)],       1, [(0,2),(4,2)]),
    ([(5,2),(0,2)],       0, [(0,2),(5,2)]),
    ([],                  5, []),
    ([(3,4)],             5, [(3,4)]),
    ([(0,2),(3,2),(6,2)], 1, [(0,8)]),
]:
    got = _merge_slots(slots, gap)
    assert got == exp, f"_merge_slots({slots},{gap}): expected {exp}, got {got}"
print("  _merge_slots            OK")

uf = _UnionFind()
[uf.add(x) for x in 'abcde']
uf.union('a','b'); uf.union('b','c'); uf.union('d','e')
assert uf.find('a') == uf.find('c')
assert uf.find('d') != uf.find('a')
assert uf.find('d') == uf.find('e')
groups = uf.groups()
assert sorted(len(v) for v in groups.values()) == [2, 3]
print("  _UnionFind              OK")

# ─────────────────────────────────────────────────
# Section 2: Constraint checkers
# ─────────────────────────────────────────────────
section("2. Constraint checkers")

# MaxDays
cbt = {'c1': tb(MON,48,12), 'c2': tb(TUE,60,12)}
assert     _check_max_days(cbt, '2', NW, ND)
assert not _check_max_days(cbt, '1', NW, ND)
WED = 2
cbt3d = {'c1':tb(MON,48,12),'c2':tb(WED,48,12),'c3':tb(FRI,48,12)}
assert     _check_max_days(cbt3d, '3', NW, ND)
assert not _check_max_days(cbt3d, '2', NW, ND)
print("  _check_max_days         OK")

# MaxDayLoad
cbt_same = {'c1': tb(MON,48,12), 'c2': tb(MON,60,12)}
assert     _check_max_day_load(cbt_same, '24', NW, ND)
assert not _check_max_day_load(cbt_same, '23', NW, ND)
cbt_diff = {'c1': tb(MON,48,12), 'c2': tb(TUE,48,12)}
assert     _check_max_day_load(cbt_diff, '12', NW, ND)
assert not _check_max_day_load(cbt_diff, '11', NW, ND)
print("  _check_max_day_load     OK")

# MaxBreaks
cbt2 = {'c1': tb(MON,48,12), 'c2': tb(MON,100,12)}
assert     _check_max_breaks(cbt2, '1,6', NW, ND)
assert not _check_max_breaks(cbt2, '0,6', NW, ND)
cbt_adj = {'c1': tb(MON,48,12), 'c2': tb(MON,60,12)}
assert     _check_max_breaks(cbt_adj, '0,6', NW, ND)
cbt3b = {'c1':tb(MON,48,12),'c2':tb(MON,100,12),'c3':tb(MON,150,12)}
assert     _check_max_breaks(cbt3b, '2,6', NW, ND)
assert not _check_max_breaks(cbt3b, '1,6', NW, ND)
print("  _check_max_breaks       OK")

# MaxBlock
cbt_adj = {'c1': tb(MON,48,12), 'c2': tb(MON,60,12)}
assert     _check_max_block(cbt_adj, '24,0', NW, ND)
assert not _check_max_block(cbt_adj, '23,0', NW, ND)
cbt_gap5 = {'c1': tb(MON,48,12), 'c2': tb(MON,65,12)}
assert     _check_max_block(cbt_gap5, '29,5', NW, ND)
assert not _check_max_block(cbt_gap5, '28,5', NW, ND)
assert     _check_max_block({'c1': tb(MON,48,12)}, '1,0', NW, ND)
print("  _check_max_block        OK")

# ─────────────────────────────────────────────────
# Section 3: Instance-level tests (agh-fal17)
# ─────────────────────────────────────────────────
section("3. Instance tests — agh-fal17.xml")

reader = PSTTReader('data/source/instances/agh-fal17.xml')
decomp = SpecialConstraintDecomposer(reader)
res = decomp.classify_classes()

all_c = set(reader.classes.keys())
assert (res.fixed | res.community | res.free) == all_c
assert not (res.fixed & res.community)
assert not (res.fixed & res.free)
assert not (res.community & res.free)
print(f"  Partition OK — {len(all_c)} = {len(res.fixed)} fixed + {len(res.community)} community + {len(res.free)} free")

for comm in res.communities:
    for cid in comm.class_ids:
        assert cid in reader.classes
    for cons in comm.constraints:
        assert not (set(cons['classes']) - comm.class_ids)
        assert _is_special(cons['type'])
seen = {}
for comm in res.communities:
    for cid in comm.class_ids:
        assert cid not in seen, f"Class {cid} in two communities"
        seen[cid] = comm.id
print(f"  Community consistency OK — {len(res.communities)} communities")

fixed_a = decomp.get_fixed_assignments()
assert set(fixed_a.keys()) == res.fixed
for cid, (tidx, rid) in fixed_a.items():
    cls = reader.classes[cid]
    assert 0 <= tidx < len(cls['time_options'])
    if rid: assert rid in reader.rooms
print(f"  Fixed assignments OK — {len(fixed_a)}")

results = []
for comm in res.communities:
    a = {cid: (0, reader.classes[cid]['room_options'][0]['id']
               if reader.classes[cid]['room_options'] else None)
         for cid in comm.class_ids}
    ok = decomp.check_special_constraints(comm, a)
    assert isinstance(ok, bool)
    results.append(ok)
for comm in res.communities[:3]:
    assert decomp.check_special_constraints(comm, {}) is True
n_pass = sum(results)
print(f"  check_special_constraints OK — {n_pass}/{len(results)} communities satisfied by first-option assignment")

# ─────────────────────────────────────────────────
# Section 4: Cross-instance partition check
# ─────────────────────────────────────────────────
section("4. Cross-instance partition check")

paths = sorted(glob.glob('data/source/**/*.xml', recursive=True))
errors = []
for path in paths:
    name = os.path.basename(path)
    try:
        r = PSTTReader(path)
        d = SpecialConstraintDecomposer(r)
        res2 = d.classify_classes()
        assert (res2.fixed | res2.community | res2.free) == set(r.classes.keys())
        assert not (res2.fixed & res2.community)
        assert not (res2.fixed & res2.free)
        assert not (res2.community & res2.free)
    except Exception as e:
        errors.append(f"  {name}: {e}")

if errors:
    print("ERRORS:")
    for e in errors: print(e)
    sys.exit(1)
else:
    print(f"  All {len(paths)} instances pass partition check")

print("\n" + "="*50)
print("  ALL TESTS PASSED")
print("="*50)
