"""
Experiment 08: IC formulation — soft (weighted penalty) vs hard (exact) — head to head.
========================================================================================

The repo mixes two IC formulations: the 2D solver (qbox/pinn2d.py) uses a
SOFT IC with an arbitrary, untested weight (1000); the 1D experiments use a
HARD IC with no weight at all. This experiment settles which one is
actually better, in the same controlled 1D setting, so the choice is a
measured decision rather than an accident of which script was written first.

Setup: same architecture/epochs as exp03, at px=2 (a clear-failure regime,
so reweighting has its best chance to help). Sweep w_ic over 4 orders of
magnitude for the soft-IC variant; run the hard-IC variant as a reference,
5 seeds each.

Hypothesis: hard IC <= best soft IC in error (it satisfies the IC exactly
and removes a hyperparameter), and NO formulation/weight rescues the
dynamics -- the time-dependent failure is not a loss-balance artifact.

Decision rule: the formulation with the lower error is adopted for every
subsequent experiment (exp06, exp07, exp09) and for the 2D re-run (Task 2.2).
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common1d import PINNSolver1DTD, make_psi0, split_operator_1d, mean_relative_error, x_lo, x_hi

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

PX = 2.0
SIGMA = 1.0
SEEDS = [0, 1, 2, 3, 4]
W_IC_VALUES = [1, 10, 100, 1000, 10000]
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_one(ic, seed, w_ic, gt_frames, x_plot):
    solver = PINNSolver1DTD(px=PX, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic)
    solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS, verbose=False)
    errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
    return float(np.mean(list(errors.values())))


def main():
    print(f"Ground truth (px={PX})...")
    psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=PX)
    gt_frames = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)
    x_plot = np.linspace(x_lo, x_hi, 512)

    results = {"w_ic": {}, "hard": []}

    print("\n--- Hard IC (reference) ---")
    for seed in SEEDS:
        err = run_one("hard", seed, None, gt_frames, x_plot)
        print(f"  seed={seed} | mean error {err:.2f}%")
        results["hard"].append(err)

    print("\n--- Soft IC sweep ---")
    for w_ic in W_IC_VALUES:
        errs = []
        for seed in SEEDS:
            err = run_one("soft", seed, w_ic, gt_frames, x_plot)
            print(f"  w_ic={w_ic:<6} seed={seed} | mean error {err:.2f}%")
            errs.append(err)
        results["w_ic"][str(w_ic)] = errs

    hard_mean = float(np.mean(results["hard"]))
    hard_std = float(np.std(results["hard"]))

    soft_means = {w: float(np.mean(v)) for w, v in results["w_ic"].items()}
    soft_stds = {w: float(np.std(v)) for w, v in results["w_ic"].items()}
    best_w = min(soft_means, key=soft_means.get)
    best_soft_mean = soft_means[best_w]

    decision = "hard" if hard_mean <= best_soft_mean else "soft"

    print("\n" + "=" * 64)
    print("DECISION")
    print("=" * 64)
    print(f"Hard IC:            {hard_mean:.2f}% +- {hard_std:.2f}%")
    print(f"Best soft IC (w={best_w}): {best_soft_mean:.2f}% +- {soft_stds[best_w]:.2f}%")
    print(f"Adopted formulation: {decision.upper()} IC")
    if decision == "soft":
        print("TRIPWIRE: soft IC beat hard IC -- flag to owner before proceeding.")
    print("=" * 64)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ws = [int(w) for w in W_IC_VALUES]
    means = [soft_means[str(w)] for w in ws]
    stds = [soft_stds[str(w)] for w in ws]
    ax.errorbar(ws, means, yerr=stds, fmt='o-', color='#3498db', capsize=4,
                lw=2, label='Soft IC (weight sweep)')
    ax.axhline(hard_mean, color='#e74c3c', lw=2, ls='--',
               label=f'Hard IC reference ({hard_mean:.1f}% +- {hard_std:.1f}%)')
    ax.fill_between([min(ws), max(ws)], hard_mean - hard_std, hard_mean + hard_std,
                     color='#e74c3c', alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel('IC loss weight w_ic (log scale)')
    ax.set_ylabel('Mean relative L2 error on |psi|^2 (%)')
    ax.set_title(f'Exp 08: IC formulation comparison (px={PX}, 5 seeds)\n'
                 f'Adopted: {decision.upper()} IC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'exp08_ic_formulation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    with open(os.path.join(RESULTS_DIR, "exp08_ic_formulation.json"), "w") as f:
        json.dump({
            "px": PX, "seeds": SEEDS, "w_ic_values": W_IC_VALUES,
            "hard_errors": results["hard"], "hard_mean": hard_mean, "hard_std": hard_std,
            "soft_errors": results["w_ic"], "soft_means": soft_means, "soft_stds": soft_stds,
            "best_w_ic": best_w, "decision": decision,
        }, f, indent=2)


if __name__ == "__main__":
    main()
