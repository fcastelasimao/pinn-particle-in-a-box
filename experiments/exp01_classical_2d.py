"""
Experiment 01: Classical split-operator baseline (2D double-slit).

Thin driver around qbox.classical.QuantumSimulator. This is the ground-truth
method the rest of the project compares against.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qbox.classical import QuantumSimulator

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def main():
    sim = QuantumSimulator(N=250, L=10.0, dt=0.001)

    sim.create_potential(wall_thickness=0.1, V_wall=500)
    sim.add_circular_barrier_with_slits(
        center_x=-1.5, center_y=0,
        radius=2.0, thickness=0.2, V_barrier=100,
        slit_angle_1=-5, slit_angle_2=5, slit_width=0.05
    )

    sim.initialize_wave_packet(x0=3.75, y0=0, sigma=0.5, px=-15, py=0)

    print("Running traditional solver...")
    frames, times = sim.simulate(n_steps=1000, frame_skip=10)

    final_prob = sim.check_conservation()
    print(f"\nFinal total probability: {final_prob:.6f} (should be ~1.0)")

    save_path = os.path.join(FIGURES_DIR, "traditional_solver.gif")
    sim.animate_results(frames, times, save_path=save_path)


if __name__ == "__main__":
    main()
