"""
Experiment 10: Fourier features -- does the textbook spectral-bias fix help?
=============================================================================

exp06 established that error grows with momentum (0.19% -> 37.68% from
px=1 to px=4) and attributed it to spectral bias: plain MLPs are known to
struggle to represent high-frequency functions (Rahaman et al. 2019).
Tancik et al. 2020 and Wang et al. 2021 show that embedding the input
coordinate with sin/cos bands at multiple frequency scales before the MLP
is a standard, well-documented mitigation.

This experiment tests it directly at the worst case measured so far
(px=4, hard IC -- the adopted formulation from exp08), same architecture,
epochs, and seeds as exp06, changing only Net1D's input embedding.

Hypothesis: Fourier features reduce error at px=4 relative to the plain-MLP
baseline. Not tested: whether they help or hurt at low px, or whether a
different scale schedule does better -- out of scope for this experiment.
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

PX = 4.0
SIGMA = 1.0
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
FOURIER_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)


def main():
    ic, w_ic = get_adopted_ic()
    x_plot = np.linspace(x_lo, x_hi, 512)

    psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=PX)
    gt_frames = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)

    results = {"baseline": [], "fourier": []}
    for variant, use_fourier in [("baseline", False), ("fourier", True)]:
        print(f"\n--- variant = {variant} ---")
        for seed in SEEDS:
            solver = PINNSolver1DTD(px=PX, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic,
                                     fourier_features=use_fourier,
                                     fourier_scales=FOURIER_SCALES)
            solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS, verbose=False)
            errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
            err = float(np.mean(list(errors.values())))
            print(f"  seed={seed} | mean error {err:.2f}%")
            results[variant].append(err)

    means = {v: float(np.mean(results[v])) for v in results}
    stds = {v: float(np.std(results[v])) for v in results}

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    for v in ["baseline", "fourier"]:
        print(f"  {v:<10} mean error {means[v]:.2f}% +- {stds[v]:.2f}%")
    improved = means["fourier"] < means["baseline"]
    print(f"Fourier features improve on baseline: {improved}")
    print("=" * 64)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = ["Baseline\n(plain MLP)", "Fourier\nfeatures"]
    vals = [means["baseline"], means["fourier"]]
    errs = [stds["baseline"], stds["fourier"]]
    colors = ['#e74c3c', '#2ecc71' if improved else '#e74c3c']
    ax.bar(labels, vals, yerr=errs, capsize=6, color=colors)
    ax.set_ylabel('Mean relative L2 error on |psi|^2 (%)')
    ic_label = f'soft IC w={w_ic:g}' if ic == 'soft' else 'hard IC'
    ax.set_title(f'Exp 10: Fourier features at px={PX:g} ({ic_label}, 5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'exp10_fourier_features.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    with open(os.path.join(RESULTS_DIR, "exp10_fourier_features.json"), "w") as f:
        json.dump({
            "ic": ic, "w_ic": w_ic, "px": PX, "seeds": SEEDS,
            "fourier_scales": list(FOURIER_SCALES),
            "errors": results, "means": means, "stds": stds,
            "improved": improved,
        }, f, indent=2)


if __name__ == "__main__":
    main()
