"""
Experiment 10b: Fourier features, retuned -- was exp10's failure a bandwidth
mismatch rather than a real negative result?
=============================================================================

exp10 tested Fourier features with a wide, untuned band (1, 2, 4, 8, 16) and
found they made px=4 WORSE (65.48% vs 37.68% baseline). Tancik et al. 2020
show frequency-band scale is the most sensitive hyperparameter in this
technique -- picking a band far wider than the signal's actual frequency
content is a known way to make it hurt rather than help.

The physically motivated band for px=4: the dominant wavenumber in
psi0 = exp(i*px*x) * Gaussian(sigma=1) is k=px=4 rad/unit, with a k-space
spread of order 1/sigma=1 from the envelope. So the signal's frequency
content lives around k in [3, 5], not up to 16. This experiment retries
with scales (2, 4, 6) -- centered on px, no wasted octaves -- same 5 seeds,
same architecture and training budget as exp10, changing only the band.

Reuses exp10's baseline (already measured, not rerun). Appends a
"fourier_tuned" entry to exp10_fourier_features.json and regenerates the
figure with three bars.
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "exp10_fourier_features.json")

PX = 4.0
SIGMA = 1.0
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5000
N_COLLOCATION = 2000
FRAME_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
TUNED_SCALES = (2.0, 4.0, 6.0)


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    ic, w_ic = data["ic"], data["w_ic"]
    x_plot = np.linspace(x_lo, x_hi, 512)

    psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=PX)
    gt_frames = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=FRAME_TIMES)

    print(f"\n--- variant = fourier_tuned (scales={TUNED_SCALES}) ---")
    errs = []
    for seed in SEEDS:
        solver = PINNSolver1DTD(px=PX, sigma=SIGMA, seed=seed, ic=ic, w_ic=w_ic,
                                 fourier_features=True, fourier_scales=TUNED_SCALES)
        solver.train(n_collocation=N_COLLOCATION, epochs=EPOCHS, verbose=False)
        errors = mean_relative_error(solver, gt_frames, FRAME_TIMES, x_plot)
        err = float(np.mean(list(errors.values())))
        print(f"  seed={seed} | mean error {err:.2f}%")
        errs.append(err)

    mean_tuned = float(np.mean(errs))
    std_tuned = float(np.std(errs))

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    print(f"  baseline      mean error {data['means']['baseline']:.2f}% +- {data['stds']['baseline']:.2f}%")
    print(f"  fourier(wide) mean error {data['means']['fourier']:.2f}% +- {data['stds']['fourier']:.2f}%")
    print(f"  fourier(tuned) mean error {mean_tuned:.2f}% +- {std_tuned:.2f}%")
    improved = mean_tuned < data["means"]["baseline"]
    print(f"Tuned Fourier features improve on baseline: {improved}")
    print("=" * 64)

    data["errors"]["fourier_tuned"] = errs
    data["means"]["fourier_tuned"] = mean_tuned
    data["stds"]["fourier_tuned"] = std_tuned
    data["fourier_tuned_scales"] = list(TUNED_SCALES)
    data["improved_tuned"] = improved
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["Baseline\n(plain MLP)", "Fourier\n(wide band)", "Fourier\n(tuned band)"]
    vals = [data["means"]["baseline"], data["means"]["fourier"], mean_tuned]
    errs_plot = [data["stds"]["baseline"], data["stds"]["fourier"], std_tuned]
    colors = ['#e74c3c', '#e74c3c', '#2ecc71' if improved else '#e74c3c']
    ax.bar(labels, vals, yerr=errs_plot, capsize=6, color=colors)
    ax.set_ylabel('Mean relative L2 error on |psi|^2 (%)')
    ic_label = f'soft IC w={w_ic:g}' if ic == 'soft' else 'hard IC'
    ax.set_title(f'Exp 10: Fourier features at px={PX:g} ({ic_label}, 5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'exp10_fourier_features.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
