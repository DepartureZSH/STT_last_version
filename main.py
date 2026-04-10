"""
main.py — ITC 2019 Timetabling Solver entry point

Usage
-----
  # Pure MIP
  python main.py --instance data/source/test/bet-sum18.xml --technique MIP

  # Hybrid (random community pre-solver + MIP)
  python main.py --instance data/source/instances/agh-fal17.xml --technique hybrid

  # Override config file
  python main.py --instance ... --config my_config.yaml

  # Override output path
  python main.py --instance ... --output solutions/my_sol.xml
"""

import argparse
import logging
import os
import sys
import yaml

from src.utils.dataReader import PSTTReader


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
def _make_logger(name: str = "solver") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ITC 2019 University Timetabling Solver"
    )
    p.add_argument(
        "--instance", required=True,
        help="Path to the ITC 2019 XML instance file",
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    p.add_argument(
        "--technique", choices=["MIP", "hybrid"], default=None,
        help="Solver technique: 'MIP' or 'hybrid'. "
             "Overrides config.yaml config.technique.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output XML path (default: solutions/<instance_name>.xml)",
    )
    p.add_argument(
        "--time-limit", type=int, default=None,
        help="MIP time limit in seconds (overrides config.yaml)",
    )
    p.add_argument(
        "--num-solutions", type=int, default=10,
        help="Number of solutions to collect (default: 10, max: 100)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args   = _parse_args()
    logger = _make_logger()

    # ── Load config ──────────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.technique:
        config.setdefault("config", {})["technique"] = args.technique
    if args.time_limit:
        config.setdefault("train", {}).setdefault("MIP", {})["time_limit"] = args.time_limit

    technique = config.get("config", {}).get("technique", "MIP")

    # ── Determine output path ─────────────────────────────────────────────────
    instance_stem = os.path.splitext(os.path.basename(args.instance))[0]
    output_dir    = os.path.join("solutions", instance_stem)
    os.makedirs(output_dir, exist_ok=True)

    # ── Validate num_solutions ───────────────────────────────────────────────
    num_solutions = min(max(args.num_solutions, 1), 100)
    if args.num_solutions != num_solutions:
        logger.warning(f"Clamping num_solutions to {num_solutions}")

    # ── Load instance ─────────────────────────────────────────────────────────
    if not os.path.exists(args.instance):
        logger.error(f"Instance file not found: {args.instance}")
        return 1

    logger.info(f"Loading instance: {args.instance}")
    reader = PSTTReader(args.instance)
    logger.info(
        f"  {reader.problem_name}: {len(reader.classes)} classes, "
        f"{len(reader.rooms)} rooms"
    )

    # ── Run solver ───────────────────────────────────────────────────────────
    if technique == "hybrid":
        return _run_hybrid(reader, config, output_dir, instance_stem, num_solutions, logger)
    else:
        return _run_mip(reader, config, output_dir, instance_stem, num_solutions, logger)


def _run_mip(reader, config, output_dir, instance_stem, num_solutions, logger) -> int:
    from src.MIP.solver import MIPSolver

    logger.info("=== Pure MIP Solver ===")
    solver = MIPSolver(reader, logger, config)
    solver.build_model()
    assignments_list = solver.solve(num_solutions=num_solutions)

    if assignments_list is None or len(assignments_list) == 0:
        logger.error("MIP: no feasible solution found")
        return 1

    # Save all solutions
    for idx, assignments in enumerate(assignments_list, 1):
        output_path = os.path.join(output_dir, f"solution{idx}_{instance_stem}.xml")
        solver.save_solution(assignments, output_path, config)
        logger.info(f"Solution {idx}/{len(assignments_list)} saved to: {output_path}")

    logger.info(f"✓ {len(assignments_list)} solutions saved to: {output_dir}/")
    return 0


def _run_hybrid(reader, config, output_dir, instance_stem, num_solutions, logger) -> int:
    from src.hybrid.hybrid_solver import HybridSolver

    logger.info("=== Hybrid Solver ===")
    hybrid = HybridSolver(reader, config, logger)
    result = hybrid.solve(num_solutions=num_solutions)

    logger.info(
        f"\nHybrid result: success={result.success}, "
        f"iterations={result.iterations}, "
        f"mip_calls={result.mip_calls}, "
        f"infeasible={result.infeasible_calls}, "
        f"time={result.total_time_sec:.1f}s, "
        f"termination={result.termination}"
    )
    logger.info(f"Sampling stats: {result.community_stats}")

    if not result.success or result.assignments_list is None or len(result.assignments_list) == 0:
        logger.error("Hybrid: no feasible solution found")
        return 1

    # Save all solutions
    for idx, assignments in enumerate(result.assignments_list, 1):
        output_path = os.path.join(output_dir, f"solution{idx}_{instance_stem}.xml")
        hybrid.save_solution_direct(assignments, output_path)
        logger.info(f"Solution {idx}/{len(result.assignments_list)} saved to: {output_path}")

    logger.info(f"✓ {len(result.assignments_list)} solutions saved to: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
