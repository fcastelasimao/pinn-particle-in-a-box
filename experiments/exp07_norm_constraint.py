"""
Experiment 07: Test mechanism A -- amplitude decay (unitarity/norm constraint).
=================================================================================

psi = 0 trivially satisfies the PDE residual, so time-dependent PINNs can let
the amplitude decay away from the t=0 anchor. Physics says the norm is
conserved: integral |psi(x,t)|^2 dx = 1 for all t. This experiment tests
whether enforcing that fixes the dynamics failure.

Setup: reuse the exp03/exp06 machinery (IC formulation adopted by exp08).
Compare baseline vs +norm-constraint (weight lambda=1.0, headline) at
px in {1, 4}, 5 seeds each. Small lambda-sensitivity check: rerun the
constrained variant at px=4 with lambda in {0.1, 10}, 3 seeds each -- NOT a
grid search, just a check that the conclusion doesn't hinge on lambda.

Hypothesis: the constraint prevents amplitude collapse (N(t) stays ~1) but
does NOT fix the dispersion/phase error at high momentum -- spectral bias is
about representing fast oscillations, not amplitude. This experiment should
separate the failure into two distinct modes.
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGMA = 1.0
SEEDS = [0, 1, 2, 3, 4]
SENS_SEEDS = [0, 1, 2]
PX_VALUES = [1.0, 4.0]
LAMBDA_HEADLINE = 1.0
LAMBDA_SENSITIVITY = [0.1, 10.0]
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def train_and_score(ic, w_ic, px, seed, norm_weight, gt_frames, x_plot):
    solver = PINNSolver1DTD(px=px, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic)
    solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS,
                 norm_weight=norm_weight, verbose=False)
    errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
    err = float(np.mean(list(errors.values())))
    return solver, err


def main():
    ic, w_ic = get_adopted_ic()
    x_plot = np.linspace(x_lo, x_hi, 512)
    t_curve = np.linspace(0, 1.0, 11)

    gt = {}
    for px in PX_VALUES:
        psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=px)
        gt[px] = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)

    results = {"headline": {}, "sensitivity": {}}
    seed0_solvers = {}  # for plotting

    for px in PX_VALUES:
        print(f"\n=== px = {px} ===")
        base_errs, cons_errs = [], []
        for seed in SEEDS:
            _, e = train_and_score(ic, w_ic, px, seed, 0.0, gt[px], x_plot)
            print(f"  baseline    seed={seed} | err {e:.2f}%")
            base_errs.append(e)
        for seed in SEEDS:
            solver, e = train_and_score(ic, w_ic, px, seed, LAMBDA_HEADLINE, gt[px], x_plot)
            print(f"  constrained seed={seed} lambda={LAMBDA_HEADLINE} | err {e:.2f}%")
            cons_errs.append(e)
            if seed == 0:
                seed0_solvers[px] = solver
        results["headline"][px] = {"baseline": base_errs, "constrained": cons_errs}

    # Baseline solver seed=0 too, for the snapshot/N(t) figure
    baseline_seed0 = {}
    for px in PX_VALUES:
        solver, _ = train_and_score(ic, w_ic, px, 0, 0.0, gt[px], x_plot)
        baseline_seed0[px] = solver

    # Lambda sensitivity at px=4
    print("\n=== Lambda sensitivity (px=4) ===")
    sens = {}
    for lam in LAMBDA_SENSITIVITY:
        errs = []
        for seed in SENS_SEEDS:
            _, e = train_and_score(ic, w_ic, 4.0, seed, lam, gt[4.0], x_plot)
            print(f"  lambda={lam} seed={seed} | err {e:.2f}%")
            errs.append(e)
        sens[lam] = errs
    results["sensitivity"] = sens

    # --- Summary ---
    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    summary = {}
    for px in PX_VALUES:
        b = results["headline"][px]["baseline"]
        c = results["headline"][px]["constrained"]
        b_mean, b_std = float(np.mean(b)), float(np.std(b))
        c_mean, c_std = float(np.mean(c)), float(np.std(c))
        print(f"  px={px}: baseline {b_mean:.2f}%+-{b_std:.2f}% | "
              f"constrained {c_mean:.2f}%+-{c_std:.2f}%")
        summary[px] = dict(baseline_mean=b_mean, baseline_std=b_std,
                            constrained_mean=c_mean, constrained_std=c_std)
    lam_summary = {lam: (float(np.mean(v)), float(np.std(v))) for lam, v in sens.items()}
    print(f"  lambda sensitivity (px=4): {lam_summary}")
    print("=" * 64)

    # --- Figure: 2 rows (px) x 2 cols (snapshot @ t=1.0, N(t) curve) ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, px in enumerate(PX_VALUES):
        x_dense = np.linspace(x_lo, x_hi, 512)
        _, prob_gt, _ = gt[px][1.0]
        prob_gt_i = np.interp(x_dense, gt[px][1.0][0], prob_gt)
        prob_base = baseline_seed0[px].predict(x_dense, 1.0)
        prob_cons = seed0_solvers[px].predict(x_dense, 1.0)

        ax = axes[row][0]
        ax.plot(x_dense, prob_gt_i, 'k-', lw=2.5, alpha=0.5, label='Truth')
        ax.plot(x_dense, prob_base, 'r--', lw=2, label='Baseline')
        ax.plot(x_dense, prob_cons, 'g--', lw=2, label='+norm constraint')
        ax.set_title(f'px={px}: |psi|^2 at t=1.0', fontsize=11, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('|psi|^2')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        N_base = baseline_seed0[px].norm_curve(t_curve)
        N_cons = seed0_solvers[px].norm_curve(t_curve)
        ax2 = axes[row][1]
        ax2.plot(t_curve, N_base, 'r--', lw=2, marker='o', ms=4, label='Baseline')
        ax2.plot(t_curve, N_cons, 'g--', lw=2, marker='o', ms=4, label='+norm constraint')
        ax2.axhline(1.0, color='k', lw=1, ls=':', alpha=0.6, label='N=1 (conserved)')
        ax2.set_title(f'px={px}: N(t) = integral |psi|^2 dx', fontsize=11, fontweight='bold')
        ax2.set_xlabel('t'); ax2.set_ylabel('N(t)')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    ic_label = f'soft IC w={w_ic:g}' if ic == 'soft' else 'hard IC'
    plt.suptitle(f'Exp 07: Norm/unitarity constraint ({ic_label}, lambda={LAMBDA_HEADLINE})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(FIGURES_DIR, 'exp07_norm_constraint.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    with open(os.path.join(RESULTS_DIR, "exp07_norm_constraint.json"), "w") as f:
        json.dump({
            "ic": ic, "w_ic": w_ic, "px_values": PX_VALUES, "seeds": SEEDS, "lambda": LAMBDA_HEADLINE,
            "headline": {str(px): results["headline"][px] for px in PX_VALUES},
            "summary": {str(px): summary[px] for px in PX_VALUES},
            "sensitivity_seeds": SENS_SEEDS,
            "sensitivity": {str(lam): v for lam, v in sens.items()},
            "sensitivity_summary": {str(lam): lam_summary[lam] for lam in LAMBDA_SENSITIVITY},
        }, f, indent=2)


if __name__ == "__main__":
    main()
