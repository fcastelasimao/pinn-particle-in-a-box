"""
Comparison of Traditional Solver vs PINN
=========================================

This script compares the split-operator method (ground truth) with the 
Physics-Informed Neural Network approach for solving the 2D Schrödinger equation.

Metrics compared:
- Accuracy (L2 error, relative error)
- Speed (training time, inference time)
- Physics conservation (probability normalization)
- Visual differences (side-by-side, error maps)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time
import torch

from traditional_solver import QuantumSimulator
from pinn_solver import PINNSolver


class SolverComparison:
    """
    Class to compare traditional solver and PINN.
    """
    
    def __init__(self, N=100, L=10.0, dt=0.001):
        """
        Initialize comparison with common parameters.
        
        Parameters:
        -----------
        N : int
            Grid size (use smaller for PINN, e.g., 100)
        L : float
            Box size
        dt : float
            Time step
        """
        self.N = N
        self.L = L
        self.dt = dt
        
        # Results will be stored here
        self.frames_traditional = None
        self.frames_pinn = None
        self.times = None
        
        # Timing data
        self.time_traditional = None
        self.time_pinn_train = None
        self.time_pinn_inference = None
        
        # Simulator instances
        self.sim_traditional = None
        self.pinn_solver = None
    
    def setup_problem(self):
        """
        Set up the quantum problem (initial condition and potential).
        Same setup for both solvers.
        """
        print("\n" + "="*60)
        print("Setting up quantum problem...")
        print("="*60)
        
        # Create traditional simulator
        self.sim_traditional = QuantumSimulator(N=self.N, L=self.L, dt=self.dt)
        
        # Set up potential
        self.sim_traditional.create_potential(wall_thickness=0.1, V_wall=500)
        self.sim_traditional.add_circular_barrier_with_slits(
            center_x=-1.5, center_y=0,
            radius=2.0, thickness=0.2, V_barrier=100,
            slit_angle_1=-5, slit_angle_2=5, slit_width=0.05
        )
        
        # Initialize wave packet
        psi_initial = self.sim_traditional.initialize_wave_packet(
            x0=3.75, y0=0, sigma=0.5, px=-15, py=0
        )
        
        print(f"✓ Grid size: {self.N}×{self.N}")
        print(f"✓ Box size: {self.L}")
        print(f"✓ Initial wave packet created")
        print(f"✓ Circular barrier with 2 slits added")
        
        return psi_initial
    
    def run_traditional(self, n_steps=1000, frame_skip=10):
        """
        Run the traditional split-operator solver.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        frame_skip : int
            Save every frame_skip-th frame
        
        Returns:
        --------
        frames, times : lists
            Probability densities and corresponding times
        """
        print("\n" + "="*60)
        print("Running TRADITIONAL SOLVER (Split-Operator Method)")
        print("="*60)
        
        start_time = time.time()
        self.frames_traditional, self.times = self.sim_traditional.simulate(
            n_steps=n_steps, frame_skip=frame_skip
        )
        self.time_traditional = time.time() - start_time
        
        # Check conservation
        final_prob = self.sim_traditional.check_conservation()
        
        print(f"\n✓ Simulation complete!")
        print(f"✓ Time taken: {self.time_traditional:.2f} seconds")
        print(f"✓ Total probability: {final_prob:.6f} (should be ~1.0)")
        print(f"✓ Frames generated: {len(self.frames_traditional)}")
        
        return self.frames_traditional, self.times
    
    def run_pinn(self, psi_initial, t_max=1.0, epochs=2000, 
                n_collocation=2000, n_initial=500):
        """
        Run the PINN solver.
        
        Parameters:
        -----------
        psi_initial : 2D complex array
            Initial wave function
        t_max : float
            Maximum time to simulate
        epochs : int
            Training epochs
        n_collocation : int
            Collocation points for PDE
        n_initial : int
            Points for initial condition
        
        Returns:
        --------
        frames : list
            Probability densities
        """
        print("\n" + "="*60)
        print("Running PINN SOLVER (Physics-Informed Neural Network)")
        print("="*60)
        
        # Create PINN
        self.pinn_solver = PINNSolver(
            self.sim_traditional.X, 
            self.sim_traditional.Y, 
            self.sim_traditional.V
        )
        
        # Train
        start_train = time.time()
        self.pinn_solver.train(
            psi_initial, 
            t_max=t_max, 
            epochs=epochs,
            n_collocation=n_collocation,
            n_initial=n_initial
        )
        self.time_pinn_train = time.time() - start_train
        
        # Inference
        start_inference = time.time()
        self.frames_pinn = self.pinn_solver.predict(self.times)
        self.time_pinn_inference = time.time() - start_inference
        
        print(f"\n✓ PINN training complete!")
        print(f"✓ Training time: {self.time_pinn_train:.2f} seconds")
        print(f"✓ Inference time: {self.time_pinn_inference:.2f} seconds")
        print(f"✓ Frames generated: {len(self.frames_pinn)}")
        
        return self.frames_pinn
    
    def compute_errors(self):
        """
        Compute various error metrics between traditional and PINN.
        
        Returns:
        --------
        metrics : dict
            Dictionary of error metrics
        """
        print("\n" + "="*60)
        print("Computing Error Metrics")
        print("="*60)
        
        if self.frames_traditional is None or self.frames_pinn is None:
            raise ValueError("Must run both solvers first!")
        
        n_frames = len(self.frames_traditional)
        
        # Initialize arrays for per-frame metrics
        l2_errors = np.zeros(n_frames)
        relative_errors = np.zeros(n_frames)
        max_errors = np.zeros(n_frames)
        
        for i in range(n_frames):
            trad = self.frames_traditional[i]
            pinn = self.frames_pinn[i]
            
            # L2 error (root mean square)
            l2_errors[i] = np.sqrt(np.mean((trad - pinn)**2))
            
            # Relative error
            relative_errors[i] = l2_errors[i] / (np.sqrt(np.mean(trad**2)) + 1e-10)
            
            # Maximum pointwise error
            max_errors[i] = np.max(np.abs(trad - pinn))
        
        # Aggregate metrics
        metrics = {
            'l2_errors': l2_errors,
            'relative_errors': relative_errors,
            'max_errors': max_errors,
            'mean_l2_error': np.mean(l2_errors),
            'mean_relative_error': np.mean(relative_errors),
            'mean_max_error': np.mean(max_errors),
            'final_l2_error': l2_errors[-1],
            'final_relative_error': relative_errors[-1]
        }
        
        # Print summary
        print(f"\n📊 Error Metrics Summary:")
        print(f"   Mean L2 Error:       {metrics['mean_l2_error']:.6f}")
        print(f"   Mean Relative Error: {metrics['mean_relative_error']*100:.2f}%")
        print(f"   Final L2 Error:      {metrics['final_l2_error']:.6f}")
        print(f"   Final Relative Error: {metrics['final_relative_error']*100:.2f}%")
        print(f"   Mean Max Error:      {metrics['mean_max_error']:.6f}")
        
        return metrics
    
    def plot_timing_comparison(self):
        """
        Create a bar chart comparing computation times.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Traditional\n(Total)', 'PINN\n(Training)', 'PINN\n(Inference)', 'PINN\n(Total)']
        times = [
            self.time_traditional,
            self.time_pinn_train,
            self.time_pinn_inference,
            self.time_pinn_train + self.time_pinn_inference
        ]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
        
        bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('timing_comparison.png', dpi=150)
        plt.show()
        
        print("\n✓ Timing comparison saved as 'timing_comparison.png'")
    
    def plot_error_evolution(self, metrics):
        """
        Plot how error evolves over time.
        
        Parameters:
        -----------
        metrics : dict
            Error metrics from compute_errors()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # L2 error over time
        ax1.plot(self.times, metrics['l2_errors'], 'r-', linewidth=2, label='L2 Error')
        ax1.fill_between(self.times, 0, metrics['l2_errors'], alpha=0.3, color='red')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('L2 Error', fontsize=12)
        ax1.set_title('L2 Error vs Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Relative error over time
        ax2.plot(self.times, metrics['relative_errors']*100, 'b-', linewidth=2, label='Relative Error')
        ax2.fill_between(self.times, 0, metrics['relative_errors']*100, alpha=0.3, color='blue')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Relative Error (%)', fontsize=12)
        ax2.set_title('Relative Error vs Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('error_evolution.png', dpi=150)
        plt.show()
        
        print("✓ Error evolution saved as 'error_evolution.png'")
    
    def create_comparison_animation(self):
        """
        Create side-by-side animation: Traditional | PINN | Error
        """
        print("\n" + "="*60)
        print("Creating comparison animation...")
        print("="*60)
        
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Find common color scale
        vmax = max([f.max() for f in self.frames_traditional])
        
        # Initialize plots
        im1 = ax1.imshow(self.frames_traditional[0], 
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='plasma', vmin=0, vmax=vmax, origin='lower')
        ax1.set_title('Traditional Solver', fontsize=13, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='|ψ|²')
        
        im2 = ax2.imshow(self.frames_pinn[0],
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='plasma', vmin=0, vmax=vmax, origin='lower')
        ax2.set_title('PINN Solver', fontsize=13, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='|ψ|²')
        
        # Error map (separate color scale)
        error_frame = np.abs(self.frames_traditional[0] - self.frames_pinn[0])
        error_max = max([np.abs(self.frames_traditional[i] - self.frames_pinn[i]).max() 
                        for i in range(len(self.frames_traditional))])
        im3 = ax3.imshow(error_frame,
                        extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                        cmap='hot', vmin=0, vmax=error_max, origin='lower')
        ax3.set_title('Absolute Error', fontsize=13, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, label='|Traditional - PINN|')
        
        # Add potential overlay to all
        barrier_mask = np.where(self.sim_traditional.V > 0, 1.0, 0.0)
        for ax in [ax1, ax2, ax3]:
            ax.contour(barrier_mask, extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                      levels=[0.5], colors=['white'], linewidths=1.5, 
                      origin='lower', alpha=0.6)
        
        # Time text
        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, 
                            fontweight='bold',
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
        
        print("Saving animation as 'comparison.gif' (this may take a minute)...")
        anim.save('comparison.gif', writer='pillow', fps=10)
        print("✓ Animation saved!")
        
        plt.show()
    
    def generate_full_report(self):
        """
        Run complete comparison and generate all plots and metrics.
        """
        print("\n" + "="*70)
        print(" "*15 + "FULL COMPARISON REPORT")
        print("="*70)
        
        # Setup
        psi_initial = self.setup_problem()
        
        # Run traditional solver
        self.run_traditional(n_steps=1000, frame_skip=20)
        
        # Run PINN
        self.run_pinn(psi_initial, t_max=self.times[-1], epochs=2000)
        
        # Compute errors
        metrics = self.compute_errors()
        
        # Generate plots
        print("\n" + "="*60)
        print("Generating comparison plots...")
        print("="*60)
        
        self.plot_timing_comparison()
        self.plot_error_evolution(metrics)
        
        # Plot PINN loss
        if self.pinn_solver is not None:
            self.pinn_solver.plot_loss()
        
        # Create comparison animation
        self.create_comparison_animation()
        
        print("\n" + "="*70)
        print("✓ COMPARISON COMPLETE!")
        print("="*70)
        print("\nFiles generated:")
        print("  • timing_comparison.png")
        print("  • error_evolution.png")
        print("  • pinn_loss.png")
        print("  • comparison.gif")
        print("\n" + "="*70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create comparison object
    comparison = SolverComparison(N=100, L=10.0, dt=0.001)
    
    # Run full comparison
    comparison.generate_full_report()