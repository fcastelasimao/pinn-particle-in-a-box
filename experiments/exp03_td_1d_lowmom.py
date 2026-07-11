"""
Experiment 03: 1D Time-Dependent Schrodinger Equation
======================================================
Hard initial condition formulation, low momentum px=1.

Problem:
    i*hbar d(psi)/dt = -hbar^2/2m d^2(psi)/dx^2,   x in [-5, 5], t in [0, 1]
    psi(x, 0) = psi0(x)    (Gaussian wave packet, px=1, sigma=1)
    psi(+-5, t) = 0        (hard walls)

Hard initial condition: psi(x,t) = psi0(x) + t*net(x,t). At t=0 the IC is
satisfied exactly; the network only has to learn the deviation from psi0.
See experiments/_common1d.py for the shared implementation.

Ground truth: 1D split-operator method (same method as qbox.classical, but
reduced to 1D for speed). We compare PINN vs split-operator at several time
snapshots.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common1d import (PINNSolver1DTD, make_psi0, split_operator_1d,
                        mean_relative_error, x_lo, x_hi)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

PX = 1.0
SIGMA = 1.0


def run():
    print("Running split-operator ground truth...")
    frame_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    psi0_func = make_psi0(x0=0.0, sigma=SIGMA, px=PX)
    gt_frames = split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=frame_times)
    print("Ground truth done.\n")

    solver = PINNSolver1DTD(px=PX, sigma=SIGMA, seed=0, ic="hard")
    print(f"Training: hard IC | px={PX}, sigma={SIGMA}")
    solver.train(n_collocation=2000, epochs=5000)

    x_plot = np.linspace(x_lo, x_hi, 512)
    errors = mean_relative_error(solver, gt_frames, frame_times, x_plot)
    for t_val, rel in errors.items():
        print(f"  t={t_val:.2f} | Relative L2 error on |psi|^2: {rel:.2f}%")
    mean_err = np.mean(list(errors.values()))
    print(f"\nMean relative L2 error: {mean_err:.2f}%")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, t_val in enumerate(frame_times[:-1]):
        ax = axes[idx // 3][idx % 3]
        prob_pinn = solver.predict(x_plot, t_val)
        _, prob_gt, _ = gt_frames[t_val]
        prob_gt_i = np.interp(x_plot, gt_frames[t_val][0], prob_gt)

        ax.plot(x_plot, prob_gt_i, 'b-', lw=2, label='Split-operator (truth)')
        ax.plot(x_plot, prob_pinn, 'r--', lw=2, label=f'PINN (err: {errors[t_val]:.1f}%)')
        ax.set_title(f't = {t_val:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('|psi|^2')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1][2]
    ax.semilogy(solver.loss_history, color='green', lw=1.2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('PDE Loss')
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Exp 03: 1D Time-Dependent PINN | Hard IC | px={PX}, sigma={SIGMA} | '
        f'Mean error: {mean_err:.2f}%',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, 'exp03_td_1d_lowmom.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    run()
