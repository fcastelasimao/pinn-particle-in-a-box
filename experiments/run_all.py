"""
Run experiments 03-09 (skips the slow 2D ones: exp01, exp02).

exp08 must run before exp06/exp07/exp09 (it decides the IC formulation they
adopt) -- this script enforces that order.

To reproduce the 2D headline result (slow, minutes+):
    uv run experiments/exp02_pinn2d_comparison.py --formulation hard --epochs 2000
    uv run experiments/exp02_pinn2d_comparison.py --formulation soft --epochs 2000

To reproduce the classical baseline gif:
    uv run experiments/exp01_classical_2d.py

Multi-seed headline experiments (exp04, exp05) are run here for seed 0 only
(matches the committed figures); re-run with --seed 1..4 for the error bars
in RESULTS.md.
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = [
    "exp03_td_1d_lowmom.py",
    "exp04_eigenstates.py",
    "exp05_inverse_potential.py",
    "exp08_ic_formulation.py",
    "exp06_momentum_sweep.py",
    "exp07_norm_constraint.py",
    "exp09_causal_training.py",
]


def main():
    for script in ORDER:
        path = os.path.join(HERE, script)
        print("\n" + "#" * 70)
        print(f"# Running {script}")
        print("#" * 70)
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"FAILED: {script} (exit code {result.returncode})")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
