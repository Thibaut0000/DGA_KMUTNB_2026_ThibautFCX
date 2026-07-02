"""Regenerate every paper number, table and figure from scratch.

    python run_all.py            # full pipeline (~10-20 min on CPU)
    python run_all.py --fast     # skip the slowest steps (SD-CAE seed sweep)

Runs the scripts in dependency order, stops on the first failure, and prints a
timing summary. This is the "code available" entry point: after it finishes,
results/tables and results/figures contain everything the paper cites.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS: list[tuple[str, bool]] = [
    # (script, slow)
    ("scripts/prepare_data.py", False),
    ("scripts/run_sdcae_ablation.py", True),          # 5 seeds x 2 dims x 5 variants
    ("scripts/run_representation_baselines.py", False),
    ("scripts/run_clr_robustness.py", False),
    ("scripts/run_health_ranking.py", False),
    ("scripts/run_health_comparison.py", False),
    ("scripts/run_label_confound.py", False),
    ("scripts/run_validation_statistics.py", False),
    ("scripts/run_chemistry_blocked_stats.py", False),
    ("scripts/run_thai_label_extension.py", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the slowest steps")
    args = ap.parse_args()

    timings: list[tuple[str, float, str]] = []
    for script, slow in STEPS:
        if args.fast and slow:
            timings.append((script, 0.0, "skipped (--fast)"))
            continue
        t0 = time.time()
        print(f"\n=== {script} " + "=" * max(1, 60 - len(script)))
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        timings.append((script, dt, status))
        if r.returncode != 0:
            break

    print("\n" + "=" * 72)
    print(f"{'script':45s} {'time':>8s}  status")
    for script, dt, status in timings:
        print(f"{script:45s} {dt:7.1f}s  {status}")
    failed = any("FAILED" in s for _, _, s in timings)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
