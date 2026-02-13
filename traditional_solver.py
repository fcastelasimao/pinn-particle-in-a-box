"""
Traditional Quantum Wave Solver using Split-Operator Method
============================================================
This module provides the ground truth simulation using the spectral 
split-operator method for solving the 2D time-dependent Schrödinger equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch


class QuantumSimulator:
    """
    2D quantum wave propagation simulator using split-operator method.
    """
    
    def __init__(self, N=250, L=10.0, dt=0.001, hbar=1.0, m=1.0):
        """
        Initialize the simulator with grid and physical parameters.
        
        Parameters:
        -----------
        N : int
            Number of grid points in each direction
        L : float
            Size of the box (from -L/2 to L/2)
        dt : float
            Time step
        hbar : float
            Reduced Planck constant (natural units)
        m : float
            Particle mass (natural units)
        """
        self.N = N
        self.L = L
        self.dt = dt
        self.hbar = hbar
        self.m = m
        
        # Create spatial grid
        self.dx = L / N
        self.x = np.linspace(-L/2, L/2, N)
        self.y = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create momentum grid (for FFT)
        kx = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K = hbar**2 * (self.KX**2 + self.KY**2) / (2 * m)
        
        # Initialize potential and wave function (will be set later)
        self.V = None
        self.psi = None
    
    def initialize_wave_packet(self, x0=0, y0=0, sigma=0.5, px=2, py=0):
        """
        Create a Gaussian wave packet.
        
        Parameters:
        -----------
        x0, y0 : float
            Center position
        sigma : float
            Width of the wave packet
        px, py : float
            Momentum in x and y directions
        
        Returns:
        --------
        psi : 2D complex array
            Initial wave function
        """
        gaussian = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))
        plane_wave = np.exp(1j * (px * self.X + py * self.Y) / self.hbar)
        psi = gaussian * plane_wave
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dx)
        psi = psi / norm
        
        self.psi = psi
        return psi
    
    def create_potential(self, wall_thickness=0.3, V_wall=500):
        """
        Create potential walls at boundaries.
        
        Parameters:
        -----------
        wall_thickness : float
            Thickness of walls
        V_wall : float
            Height of wall potential
        
        Returns:
        --------
        V : 2D array
            Potential landscape
        """
        V = np.zeros_like(self.X)
        
        # Boundary walls
        V[self.X < -self.L/2 + wall_thickness] = V_wall
        V[self.X > self.L/2 - wall_thickness] = V_wall
        V[self.Y < -self.L/2 + wall_thickness] = V_wall
        V[self.Y > self.L/2 - wall_thickness] = V_wall
        
        self.V = V
        return V
    
    def add_circular_barrier_with_slits(self, center_x=0, center_y=0,
                                        radius=2.0, thickness=0.2,
                                        V_barrier=100,
                                        slit_angle_1=0, slit_angle_2=None,
                                        slit_width=0.3):
        """
        Add a circular barrier with slits to the potential.
        
        Parameters:
        -----------
        center_x, center_y : float
            Center of the circle
        radius : float
            Circle radius
        thickness : float
            Barrier thickness
        V_barrier : float
            Barrier height
        slit_angle_1, slit_angle_2 : float
            Angles of slits in degrees
        slit_width : float
            Angular width of slits in radians
        """
        if self.V is None:
            self.V = np.zeros_like(self.X)
        
        r = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        theta = np.arctan2(self.Y - center_y, self.X - center_x)
        
        barrier_mask = (r >= radius) & (r <= radius + thickness)
        self.V[barrier_mask] = V_barrier
        
        # Cut out slits
        angle_1_rad = np.deg2rad(slit_angle_1)
        slit_1_mask = barrier_mask & (np.abs(theta - angle_1_rad) < slit_width / 2)
        self.V[slit_1_mask] = 0
        
        if slit_angle_2 is not None:
            angle_2_rad = np.deg2rad(slit_angle_2)
            slit_2_mask = barrier_mask & (np.abs(theta - angle_2_rad) < slit_width / 2)
            self.V[slit_2_mask] = 0
    
    def step(self):
        """
        Evolve wave function by one time step using split-operator method.
        """
        if self.psi is None or self.V is None:
            raise ValueError("Must initialize wave function and potential first")
        
        # Half-step potential
        self.psi = self.psi * np.exp(-1j * self.V * self.dt / (2 * self.hbar))
        
        # Full-step kinetic (in Fourier space)
        psi_k = np.fft.fft2(self.psi)
        psi_k = psi_k * np.exp(-1j * self.K * self.dt / self.hbar)
        self.psi = np.fft.ifft2(psi_k)
        
        # Half-step potential
        self.psi = self.psi * np.exp(-1j * self.V * self.dt / (2 * self.hbar))
    
    def simulate(self, n_steps, frame_skip=10):
        """
        Run the simulation for n_steps time steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        frame_skip : int
            Save every frame_skip-th frame
        
        Returns:
        --------
        frames : list of 2D arrays
            Probability density at each saved time
        times : list of floats
            Time values for each frame
        """
        frames = []
        times = []
        
        for step in range(n_steps):
            self.step()
            
            if step % frame_skip == 0:
                probability = np.abs(self.psi)**2
                frames.append(probability.copy())
                times.append(step * self.dt)
            
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}")
        
        return frames, times
    
    def check_conservation(self):
        """Check if probability is conserved."""
        total_prob = np.sum(np.abs(self.psi)**2) * self.dx * self.dx
        return total_prob
    
    def animate_results(self, frames, times, save_path='quantum_wave.gif', 
                       show_potential=True, cmap='plasma'):
        """
        Create animation of the results.
        
        Parameters:
        -----------
        frames : list of 2D arrays
            Probability densities
        times : list of floats
            Time values
        save_path : str
            Where to save the animation
        show_potential : bool
            Whether to overlay the potential
        cmap : str
            Colormap name
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        vmax = max([frame.max() for frame in frames])
        
        im = ax.imshow(frames[0], extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                      cmap=cmap, vmin=0, vmax=vmax, origin='lower')
        
        # Overlay potential if requested
        if show_potential and self.V is not None:
            barrier_mask = np.where(self.V > 0, 1.0, 0.0)
            ax.contourf(barrier_mask, extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                       levels=[0.5, 1.5], colors=['black'], alpha=0.7, origin='lower')
            ax.contour(barrier_mask, extent=[-self.L/2, self.L/2, -self.L/2, self.L/2],
                      levels=[0.5], colors=['black'], linewidths=0.8, origin='lower')
            
            legend_elements = [Patch(facecolor='black', edgecolor='black', 
                                    alpha=0.7, label='Potential Barrier')]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Quantum Wave Propagation (Traditional Solver)', fontsize=13)
        plt.colorbar(im, ax=ax, label='Probability Density |ψ|²')
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           color='black', fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        def animate(frame_idx):
            im.set_array(frames[frame_idx])
            time_text.set_text(f'Time = {times[frame_idx]:.2f}')
            return [im, time_text]
        
        anim = FuncAnimation(fig, animate, frames=len(frames),
                           interval=50, blit=False, repeat=True)
        
        print(f"Saving animation as '{save_path}'...")
        anim.save(save_path, writer='pillow', fps=20)
        print("Done!")
        
        plt.show()


# =============================================================================
# EXAMPLE USAGE / MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Create simulator
    sim = QuantumSimulator(N=250, L=10.0, dt=0.001)
    
    # Set up potential
    sim.create_potential(wall_thickness=0.1, V_wall=500)
    sim.add_circular_barrier_with_slits(
        center_x=-1.5, center_y=0,
        radius=2.0, thickness=0.2, V_barrier=100,
        slit_angle_1=-5, slit_angle_2=5, slit_width=0.05
    )
    
    # Initialize wave packet
    sim.initialize_wave_packet(x0=3.75, y0=0, sigma=0.5, px=-15, py=0)
    
    # Run simulation
    print("Running traditional solver...")
    frames, times = sim.simulate(n_steps=1000, frame_skip=10)
    
    # Check physics
    final_prob = sim.check_conservation()
    print(f"\nFinal total probability: {final_prob:.6f} (should be ~1.0)")
    
    # Create animation
    sim.animate_results(frames, times, save_path='traditional_solver.gif')