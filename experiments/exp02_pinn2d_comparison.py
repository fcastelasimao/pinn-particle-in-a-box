"""
Experiment 02: Traditional solver vs 2D PINN — the headline failure.
=====================================================================

Supports both IC formulations:
  --formulation soft --w-ic 1.0     : the ADOPTED formulation (Task 2.2) --
                                       exp08_ic_formulation.py found soft IC
                                       with weight 1 beats every other option
                                       tested, including hard IC, in 1D.
  --formulation soft --w-ic 1000.0  : original, untested weight (default) --
                                       kept as a comparison point.
  --formulation hard                : psi = psi0 + t*net -- kept available
                                       for reference, but NOT adopted (the
                                       1D investigation found it converges to
                                       a non-propagating attractor; see
                                       results.ipynb, Task 2.1).

Usage:
  python experiments/exp02_pinn2d_comparison.py --formulation soft --w-ic 1.0 --epochs 2000
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qbox.classical import QuantumSimulator
from qbox.pinn2d import PINNSolver, PINNSolverHardIC

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


class SolverComparison:
    def __init__(self, N=100, L=10.0, dt=0.001, formulation="soft", w_ic=1000.0):
        self.N = N
        self.L = L
        self.dt = dt
        self.formulation = formulation
        self.w_ic = w_ic
        # label used in output filenames -- keeps soft/w=1000 and soft/w=1 runs separate
        self.run_label = formulation if formulation == "hard" else f"soft_w{w_ic:g}"

        self.frames_traditional = None
        self.frames_pinn = None
        self.times = None

        self.time_traditional = None
        self.time_pinn_train = None
        self.time_pinn_inference = None

        self.sim_traditional = None
        self.pinn_solver = None

    def setup_problem(self):
        print("\n" + "=" * 60)
        print("Setting up quantum problem...")
        print("=" * 60)

        self.sim_traditional = QuantumSimulator(N=self.N, L=self.L, dt=self.dt)
        self.sim_traditional.create_potential(wall_thickness=0.1, V_wall=500)
        self.sim_traditional.add_circular_barrier_with_slits(
            center_x=-1.5, center_y=0,
            radius=2.0, thickness=0.2, V_barrier=100,
            slit_angle_1=-5, slit_angle_2=5, slit_width=0.05
        )
        self.wave_packet_params = dict(x0=3.75, y0=0, sigma=0.5, px=-15, py=0)
        psi_initial = self.sim_traditional.initialize_wave_packet(**self.wave_packet_params)
        print(f"Grid size: {self.N}x{self.N} | Box size: {self.L} | formulation={self.formulation}")
        return psi_initial

    def run_traditional(self, n_steps=1000, frame_skip=10):
        print("\nRunning TRADITIONAL SOLVER (Split-Operator Method)")
        start_time = time.time()
        self.frames_traditional, self.times = self.sim_traditional.simulate(
            n_steps=n_steps, frame_skip=frame_skip
        )
        self.time_traditional = time.time() - start_time
        final_prob = self.sim_traditional.check_conservation()
        print(f"Time taken: {self.time_traditional:.2f}s | Total probability: {final_prob:.6f}")
        return self.frames_traditional, self.times

    def run_pinn(self, psi_initial, t_max=1.0, epochs=2000,
                n_collocation=2000, n_initial=500):
        print(f"\nRunning PINN SOLVER ({self.formulation} IC)")

        if self.formulation == "soft":
            self.pinn_solver = PINNSolver(
                self.sim_traditional.X, self.sim_traditional.Y, self.sim_traditional.V,
                w_ic=self.w_ic
            )
            start_train = time.time()
            self.pinn_solver.train(
                psi_initial, t_max=t_max, epochs=epochs,
                n_collocation=n_collocation, n_initial=n_initial
            )
        else:
            self.pinn_solver = PINNSolverHardIC(
                self.sim_traditional.X, self.sim_traditional.Y, self.sim_traditional.V,
                **self.wave_packet_params
            )
            start_train = time.time()
            self.pinn_solver.train(
                psi_initial, t_max=t_max, epochs=epochs, n_collocation=n_collocation
            )
        self.time_pinn_train = time.time() - start_train

        start_inference = time.time()
        self.frames_pinn = self.pinn_solver.predict(self.times)
        self.time_pinn_inference = time.time() - start_inference

        print(f"Training time: {self.time_pinn_train:.2f}s | Inference: {self.time_pinn_inference:.2f}s")
        return self.frames_pinn

    def compute_errors(self):
        n_frames = len(self.frames_traditional)
        l2_errors = np.zeros(n_frames)
        relative_errors = np.zeros(n_frames)
        max_errors = np.zeros(n_frames)

        for i in range(n_frames):
            trad = self.frames_traditional[i]
            pinn = self.frames_pinn[i]
            l2_errors[i] = np.sqrt(np.mean((trad - pinn) ** 2))
            relative_errors[i] = l2_errors[i] / (np.sqrt(np.mean(trad ** 2)) + 1e-10)
            max_errors[i] = np.max(np.abs(trad - pinn))

        metrics = {
            'l2_errors': l2_errors,
            'relative_errors': relative_errors,
            'max_errors': max_errors,
            'mean_l2_error': np.mean(l2_errors),
            'mean_relative_error': np.mean(relative_errors),
            'mean_max_error': np.mean(max_errors),
            'final_l2_error': l2_errors[-1],
            'final_relative_error': relative_errors[-1],
        }
        print(f"\nMean Relative Error: {metrics['mean_relative_error']*100:.2f}%")
        print(f"Final Relative Error: {metrics['final_relative_error']*100:.2f}%")
        return metrics

    def plot_timing_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = ['Traditional\n(Total)', 'PINN\n(Training)', 'PINN\n(Inference)', 'PINN\n(Total)']
        times = [
            self.time_traditional, self.time_pinn_train, self.time_pinn_inference,
            self.time_pinn_train + self.time_pinn_inference
        ]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
        bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{time_val:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title(f'Computation Time Comparison ({self.run_label} IC)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f'timing_comparison_{self.run_label}.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

    def plot_error_evolution(self, metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.times, metrics['l2_errors'], 'r-', linewidth=2, label='L2 Error')
        ax1.fill_between(self.times, 0, metrics['l2_errors'], alpha=0.3, color='red')
        ax1.set_xlabel('Time'); ax1.set_ylabel('L2 Error')
        ax1.set_title('L2 Error vs Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(self.times, metrics['relative_errors']*100, 'b-', linewidth=2, label='Relative Error')
        ax2.fill_between(self.times, 0, metrics['relative_errors']*100, alpha=0.3, color='blue')
        ax2.set_xlabel('Time'); ax2.set_ylabel('Relative Error (%)')
        ax2.set_title(f'Relative Error vs Time ({self.run_label} IC)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3); ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f'error_evolution_{self.run_label}.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

    def create_comparison_animation(self):
        print("\nCreating comparison animation...")
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        vmax = max([f.max() for f in self.frames_traditional])

        im1 = ax1.imshow(self.frames_traditional[0],
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='plasma', vmin=0, vmax=vmax, origin='lower')
        ax1.set_title('Traditional Solver', fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='|psi|^2')

        im2 = ax2.imshow(self.frames_pinn[0],
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='plasma', vmin=0, vmax=vmax, origin='lower')
        ax2.set_title(f'PINN Solver ({self.run_label} IC)', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='|psi|^2')

        error_frame = np.abs(self.frames_traditional[0] - self.frames_pinn[0])
        error_max = max([np.abs(self.frames_traditional[i] - self.frames_pinn[i]).max()
                        for i in range(len(self.frames_traditional))])
        im3 = ax3.imshow(error_frame,
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='hot', vmin=0, vmax=error_max, origin='lower')
        ax3.set_title('Absolute Error', fontsize=13, fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='|Traditional - PINN|')

        barrier_mask = np.where(self.sim_traditional.V > 0, 1.0, 0.0)
        for ax in [ax1, ax2, ax3]:
            ax.contour(barrier_mask, extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                      levels=[0.5], colors=['white'], linewidths=1.5, origin='lower', alpha=0.6)

        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def animate(frame_idx):
            im1.set_array(self.frames_traditional[frame_idx])
            im2.set_array(self.frames_pinn[frame_idx])
            error = np.abs(self.frames_traditional[frame_idx] - self.frames_pinn[frame_idx])
            im3.set_array(error)
            time_text.set_text(f'Time = {self.times[frame_idx]:.3f}')
            return [im1, im2, im3, time_text]

        anim = FuncAnimation(fig, animate, frames=len(self.frames_traditional),
                           interval=100, blit=False, repeat=True)

        save_path = os.path.join(FIGURES_DIR, f'comparison_{self.run_label}.gif')
        print(f"Saving animation as '{save_path}'...")
        anim.save(save_path, writer='pillow', fps=10)
        print("Saved.")

    def generate_full_report(self, epochs=2000):
        psi_initial = self.setup_problem()
        self.run_traditional(n_steps=1000, frame_skip=20)
        self.run_pinn(psi_initial, t_max=self.times[-1], epochs=epochs)
        metrics = self.compute_errors()

        self.plot_timing_comparison()
        self.plot_error_evolution(metrics)
        if self.pinn_solver is not None and hasattr(self.pinn_solver, 'plot_loss'):
            self.pinn_solver.plot_loss(save_path=os.path.join(FIGURES_DIR, f'pinn_loss_{self.run_label}.png'))
        self.create_comparison_animation()

        slowdown = (self.time_pinn_train + self.time_pinn_inference) / self.time_traditional
        print("\n" + "=" * 70)
        print(f"SUMMARY ({self.run_label} IC formulation)")
        print(f"  Mean relative error: {metrics['mean_relative_error']*100:.2f}%")
        print(f"  Final relative error: {metrics['final_relative_error']*100:.2f}%")
        print(f"  Traditional time: {self.time_traditional:.2f}s")
        print(f"  PINN time (train+inference): {self.time_pinn_train + self.time_pinn_inference:.2f}s")
        print(f"  Slowdown factor: {slowdown:.1f}x")
        print("=" * 70)
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", choices=["soft", "hard"], default="soft")
    parser.add_argument("--w-ic", type=float, default=1000.0,
                         help="IC loss weight for --formulation soft. Task 2.2's adopted "
                              "run uses 1.0 (the weight exp08_ic_formulation.py found best); "
                              "the default 1000.0 reproduces the original untested weight.")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    comparison = SolverComparison(N=100, L=10.0, dt=0.001,
                                   formulation=args.formulation, w_ic=args.w_ic)
    comparison.generate_full_report(epochs=args.epochs)
