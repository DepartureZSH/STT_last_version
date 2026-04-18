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
        "--technique", choices=["MIP", "hybrid", "divide_conquer"], default=None,
        help="Solver technique: 'MIP', 'hybrid', or 'divide_conquer'. "
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
        "--num-solutions", type=int, default=None,
        help="Number of solutions to collect in the Gurobi pool "
             "(overrides config.yaml train.MIP.PoolSolutions)",
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
    if args.num_solutions:
        config.setdefault("train", {}).setdefault("MIP", {})["PoolSolutions"] = args.num_solutions

    technique = config.get("config", {}).get("technique", "MIP")

    # ── Determine output path ─────────────────────────────────────────────────
    instance_stem = os.path.splitext(os.path.basename(args.instance))[0]
    output_path   = args.output or os.path.join("solutions", f"{instance_stem}.xml")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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
        return _run_hybrid(reader, config, output_path, logger)
    elif technique == "divide_conquer":
        return _run_divide_conquer(reader, config, output_path, logger)
    else:
        return _run_mip(reader, config, output_path, logger)


def _run_mip(reader, config, output_path, logger) -> int:
    from src.MIP.solver import MIPSolver

    logger.info("=== Pure MIP Solver ===")
    solver = MIPSolver(reader, logger, config)
    solver.build_model()
    assignments_list = solver.solve()

    if assignments_list is None:
        logger.error("MIP: no feasible solution found")
        return 1

    solver.save_solution(assignments_list[0], output_path, config)
    logger.info(f"Solution saved to: {output_path}")
    return 0


def _run_hybrid(reader, config, output_path, logger) -> int:
    from src.hybrid.hybrid_solver import HybridSolver

    logger.info("=== Hybrid Solver ===")
    hybrid = HybridSolver(reader, config, logger)
    result = hybrid.solve()

    logger.info(
        f"\nHybrid result: success={result.success}, "
        f"iterations={result.iterations}, "
        f"mip_calls={result.mip_calls}, "
        f"infeasible={result.infeasible_calls}, "
        f"time={result.total_time_sec:.1f}s, "
        f"termination={result.termination}"
    )
    logger.info(f"Sampling stats: {result.community_stats}")

    if not result.success:
        logger.error("Hybrid: no feasible solution found")
        return 1

    hybrid.save_solution(result, output_path)
    logger.info(f"Solution saved to: {output_path}")
    return 0


def _run_divide_conquer(reader, config, output_path, logger) -> int:
    from src.hybrid.divide_conquer_solver import DivideConquerSolver

    logger.info("=== Divide & Conquer Solver ===")
    dc = DivideConquerSolver(reader, config, logger)
    result = dc.solve()

    logger.info(
        f"\nDivide & Conquer result: success={result.success}, "
        f"partitions={result.partitions_solved}, "
        f"mip_calls={result.total_mip_calls}, "
        f"conflicts={result.total_conflicts}, "
        f"backtracks={result.total_backtracks}, "
        f"time={result.total_time_sec:.1f}s, "
        f"termination={result.termination}"
    )

    # Save the solution regardless of strict success flag so the validator
    # can be used to inspect partial results; exit code reflects quality.
    if result.assignments:
        dc.save_solution(result, output_path)
        logger.info(f"Solution saved to: {output_path}")
    else:
        logger.error("Divide & Conquer: no assignments produced — nothing to save")
        return 1

    if not result.success:
        logger.warning(
            f"Divide & Conquer: solution has remaining issues "
            f"(violations={result.total_conflicts}, "
            f"unassigned={len(reader.classes) - len(result.assignments)})"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
