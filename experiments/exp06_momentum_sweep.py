"""
Experiment 06: Momentum sweep -- the "failure law" figure.
============================================================

exp03 showed one data point (px=1, ~46% mean error). This experiment
establishes whether error grows systematically with momentum, using the IC
formulation adopted by exp08_ic_formulation.py (run exp08 first).

Hypothesis: error increases monotonically with px, because the network must
represent phase oscillations exp(i*px*x) of increasing frequency (spectral
bias, mechanism C in plans/implementation_plan.md).

Tripwire: if the trend is NOT monotonic, that changes the story and must be
flagged to the owner rather than papered over.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common1d import (PINNSolver1DTD, make_psi0, split_operator_1d,
                        mean_relative_error, get_adopted_ic, x_lo, x_hi)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGMA = 1.0
SEEDS = [0, 1, 2, 3, 4]
PX_VALUES = [0.5, 1.0, 2.0, 4.0]
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ic, w_ic = get_adopted_ic()
    x_plot = np.linspace(x_lo, x_hi, 512)

    results = {}
    for px in PX_VALUES:
        print(f"\n--- px = {px} ---")
        psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=px)
        gt_frames = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)

        errs = []
        for seed in SEEDS:
            solver = PINNSolver1DTD(px=px, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic)
            solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS, verbose=False)
            errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
            err = float(np.mean(list(errors.values())))
            print(f"  seed={seed} | mean error {err:.2f}%")
            errs.append(err)
        results[px] = errs

    means = [float(np.mean(results[px])) for px in PX_VALUES]
    stds = [float(np.std(results[px])) for px in PX_VALUES]

    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    for px, mean, std in zip(PX_VALUES, means, stds):
        print(f"  px={px:<4} mean error {mean:.2f}% +- {std:.2f}%")
    print(f"Monotonic increase: {monotonic}")
    if not monotonic:
        print("TRIPWIRE: error vs momentum is NOT monotonic -- flag to owner.")
    print("=" * 64)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(PX_VALUES, means, yerr=stds, fmt='o-', color='#e74c3c',
                capsize=4, lw=2, ms=7)
    ax.set_xlabel('Momentum px')
    ax.set_ylabel('Mean relative L2 error on |psi|^2 (%)')
    ic_label = f'soft IC w={w_ic:g}' if ic == 'soft' else 'hard IC'
    ax.set_title(f'Exp 06: Error vs momentum ({ic_label}, 5 seeds)\n'
                 f'Monotonic increase: {monotonic}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'exp06_momentum_sweep.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    with open(os.path.join(RESULTS_DIR, "exp06_momentum_sweep.json"), "w") as f:
        json.dump({
            "ic": ic, "w_ic": w_ic, "px_values": PX_VALUES, "seeds": SEEDS,
            "errors": {str(px): results[px] for px in PX_VALUES},
            "means": means, "stds": stds, "monotonic": monotonic,
        }, f, indent=2)


if __name__ == "__main__":
    main()
