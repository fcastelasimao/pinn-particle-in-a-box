"""
Experiment 09: Test mechanism B -- causality violation (causal training).
=============================================================================

The baseline PINN samples collocation times uniformly at random: nothing in
training makes the solution at time t build on the solution before t --
only the d(psi)/dt coupling and the t=0 anchor encode temporal order. This
is the "causality violation" pathology (Wang, Sankaran & Perdikaris 2022).
The exp03 failure shape -- the PINN packet barely moves while the truth
propagates, at a momentum too low for spectral bias to explain -- makes this
the prime suspect at low px.

Causal weighting: discretise t into M=10 bins; weight bin i by
w_i = exp(-eps * sum_{j<i} L_j) with stop-gradient on the weights (built from
.detach()'d residuals); eps=10 default. See _common1d.PINNSolver1DTD.train
for the implementation.

Setup: reuse the exp03/exp06 machinery (IC formulation adopted by exp08).
Compare baseline vs causal at px in {1, 4}, 5 seeds each.

Hypothesis: causal training substantially reduces error at px=1 (failure
there is propagation-driven -- the packet doesn't move) but does NOT rescue
px=4 (failure there is representation-driven -- spectral bias). This would
cleanly separate mechanisms B and C.

Tripwire: if causal training rescues BOTH momenta, the story becomes
"causality, not spectral bias, was the dominant failure" -- flag to owner,
do not silently proceed with the original narrative.
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
PX_VALUES = [1.0, 4.0]
CAUSAL_EPS = 10.0
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def train_and_score(ic, w_ic, px, seed, causal, gt_frames, x_plot):
    solver = PINNSolver1DTD(px=px, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic)
    solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS,
                 causal=causal, causal_eps=CAUSAL_EPS, verbose=False)
    errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
    err = float(np.mean(list(errors.values())))
    return solver, err


def main():
    ic, w_ic = get_adopted_ic()
    x_plot = np.linspace(x_lo, x_hi, 512)

    gt = {}
    for px in PX_VALUES:
        psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=px)
        gt[px] = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)

    results = {}
    seed0 = {"baseline": {}, "causal": {}}

    for px in PX_VALUES:
        print(f"\n=== px = {px} ===")
        base_errs, causal_errs = [], []
        for seed in SEEDS:
            solver, e = train_and_score(ic, w_ic, px, seed, False, gt[px], x_plot)
            print(f"  baseline seed={seed} | err {e:.2f}%")
            base_errs.append(e)
            if seed == 0:
                seed0["baseline"][px] = solver
        for seed in SEEDS:
            solver, e = train_and_score(ic, w_ic, px, seed, True, gt[px], x_plot)
            print(f"  causal   seed={seed} | err {e:.2f}%")
            causal_errs.append(e)
            if seed == 0:
                seed0["causal"][px] = solver
        results[px] = {"baseline": base_errs, "causal": causal_errs}

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    summary = {}
    rescued = {}
    for px in PX_VALUES:
        b = results[px]["baseline"]; c = results[px]["causal"]
        b_mean, b_std = float(np.mean(b)), float(np.std(b))
        c_mean, c_std = float(np.mean(c)), float(np.std(c))
        improved = c_mean < 0.5 * b_mean  # "substantially reduces" threshold
        rescued[px] = improved
        print(f"  px={px}: baseline {b_mean:.2f}%+-{b_std:.2f}% | "
              f"causal {c_mean:.2f}%+-{c_std:.2f}% | substantially improved: {improved}")
        summary[px] = dict(baseline_mean=b_mean, baseline_std=b_std,
                            causal_mean=c_mean, causal_std=c_std, improved=improved)

    if rescued.get(1.0) and rescued.get(4.0):
        print("TRIPWIRE: causal training rescued BOTH momenta -- flag to owner, "
              "narrative may need to shift to causality as dominant mechanism.")
    print("=" * 64)

    # --- Figure: snapshots at t=1.0 for baseline vs causal vs truth, per px ---
    fig, axes = plt.subplots(1, len(PX_VALUES) + 1, figsize=(6 * (len(PX_VALUES) + 1), 5))
    for col, px in enumerate(PX_VALUES):
        x_dense = np.linspace(x_lo, x_hi, 512)
        _, prob_gt, _ = gt[px][1.0]
        prob_gt_i = np.interp(x_dense, gt[px][1.0][0], prob_gt)
        prob_base = seed0["baseline"][px].predict(x_dense, 1.0)
        prob_causal = seed0["causal"][px].predict(x_dense, 1.0)

        ax = axes[col]
        ax.plot(x_dense, prob_gt_i, 'k-', lw=2.5, alpha=0.5, label='Truth')
        ax.plot(x_dense, prob_base, 'r--', lw=2, label='Baseline')
        ax.plot(x_dense, prob_causal, 'b--', lw=2, label='Causal')
        ax.set_title(f'px={px} at t=1.0', fontsize=11, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('|psi|^2')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[-1]
    x_pos = np.arange(len(PX_VALUES))
    width = 0.35
    b_means = [summary[px]["baseline_mean"] for px in PX_VALUES]
    b_stds = [summary[px]["baseline_std"] for px in PX_VALUES]
    c_means = [summary[px]["causal_mean"] for px in PX_VALUES]
    c_stds = [summary[px]["causal_std"] for px in PX_VALUES]
    ax.bar(x_pos - width/2, b_means, width, yerr=b_stds, label='Baseline', color='#e74c3c', alpha=0.8, capsize=4)
    ax.bar(x_pos + width/2, c_means, width, yerr=c_stds, label='Causal', color='#3498db', alpha=0.8, capsize=4)
    ax.set_xticks(x_pos); ax.set_xticklabels([f'px={px}' for px in PX_VALUES])
    ax.set_ylabel('Mean relative L2 error (%)')
    ax.set_title('Summary (5 seeds)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    ic_label = f'soft IC w={w_ic:g}' if ic == 'soft' else 'hard IC'
    plt.suptitle(f'Exp 09: Causal training ({ic_label}, eps={CAUSAL_EPS})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(FIGURES_DIR, 'exp09_causal_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    with open(os.path.join(RESULTS_DIR, "exp09_causal_training.json"), "w") as f:
        json.dump({
            "ic": ic, "w_ic": w_ic, "px_values": PX_VALUES, "seeds": SEEDS, "causal_eps": CAUSAL_EPS,
            "results": {str(px): results[px] for px in PX_VALUES},
            "summary": {str(px): summary[px] for px in PX_VALUES},
        }, f, indent=2)


if __name__ == "__main__":
    main()
