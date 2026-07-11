"""
Experiment 04: Stationary Eigenstates -- Where the PINN Works
============================================================
Zero-momentum, time-independent eigenstates of the 1D Schrodinger equation.

WHY THIS EXPERIMENT EXISTS
--------------------------
The 2D time-dependent PINN (exp02_pinn2d_comparison.py) FAILED. The 1D
time-dependent test (exp03_td_1d_lowmom.py) showed the error GROWS with
time and momentum, and exp06_momentum_sweep.py shows it grows with momentum
specifically.

This experiment tests the OPPOSITE regime: stationary eigenstates.
  - Zero net momentum (real standing waves, no travelling phase).
  - No time evolution at all (d(psi)/dt = -iE/hbar * psi is trivial; we solve
    the time-INDEPENDENT equation H*psi = E*psi).
  - Smooth, low-frequency functions for the low-lying states.

Prediction: the SAME PINN machinery that failed on dynamics should SUCCEED
here, because we removed the variable (momentum / oscillation frequency)
that caused the failure. This pins down the boundary of where PINNs work.

METHODS COMPARED (three-way)
----------------------------
  1. Analytic      -- exact textbook solution (box and harmonic oscillator).
  2. Finite-Diff   -- the *traditional* method for stationary states: build
                     the Hamiltonian matrix H = -hbar^2/2m d^2/dx^2 + V and
                     diagonalise it with eigh(). This is the right baseline
                     here, NOT the split-operator method (that is for time
                     evolution).
  3. PINN          -- Rayleigh-quotient / Deep-Ritz minimisation. Ground
                     state from min R[psi]; excited states via orthogonal
                     deflation.

POTENTIALS
----------
  A. Infinite square well on [0, L]   (V = 0 inside, hard walls).
  B. Quantum harmonic oscillator V = 1/2 m omega^2 x^2 on a truncated domain.

Honest expectation: the PINN will MATCH analytic/FD on accuracy but will NOT
beat finite-difference on speed -- diagonalising a 1D matrix is essentially
instantaneous. The point is the accuracy regime, not a speed win. (For the
genuine "PINN beats traditional" claim, see exp05_inverse_potential.py.)

Multi-seed: pass --seed N (default 0). Only seed 0 saves the committed PNG;
all seeds append their numbers to results_raw/exp04_eigenstates.json for the
mean +- std reported in RESULTS.md (Task 2.6).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.linalg import eigh_tridiagonal
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qbox.utils import get_device, seed_everything

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()

hbar = 1.0
m = 1.0


# ===========================================================================
# 1. TRADITIONAL METHOD: finite-difference Hamiltonian diagonalisation
# ===========================================================================
def fd_eigenstates(x, V, n_states=2):
    dx = x[1] - x[0]
    kin = hbar**2 / (2 * m * dx**2)
    main_diag = 2 * kin + V
    off_diag = -kin * np.ones(len(x) - 1)
    energies, vecs = eigh_tridiagonal(main_diag, off_diag,
                                      select='i', select_range=(0, n_states - 1))
    states = []
    for k in range(n_states):
        psi = vecs[:, k]
        psi = psi / np.sqrt(np.trapz(psi**2, x))
        states.append(psi)
    return energies, np.array(states)


# ===========================================================================
# 2. PINN METHOD: Rayleigh quotient + orthogonal deflation
# ===========================================================================
class EigenNet(nn.Module):
    def __init__(self, a, b, hidden_size=64, n_layers=4):
        super().__init__()
        self.a, self.b = a, b
        self.scale = ((b - a) / 2) ** 2
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        env = (x - self.a) * (self.b - x) / self.scale
        return env * self.net(x)


def energy_functional(model, V_func, x):
    x.requires_grad_(True)
    psi = model(x)
    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                create_graph=True)[0]
    V = V_func(x)
    kinetic = (hbar**2 / (2 * m)) * torch.mean(psi_x**2)
    potential = torch.mean(V * psi**2)
    norm = torch.mean(psi**2)
    return (kinetic + potential) / norm


def train_eigenstate(V_func, a, b, prev_states=None, epochs=4000,
                     n_collocation=2000, lr=1e-3, ortho_weight=50.0):
    prev_states = prev_states or []
    model = EigenNet(a, b).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        optimizer.zero_grad()
        x = torch.rand(n_collocation, 1, device=device) * (b - a) + a
        loss = energy_functional(model, V_func, x)

        if prev_states:
            psi = model(x)
            for prev in prev_states:
                with torch.no_grad():
                    phi = prev(x)
                overlap = torch.mean(psi * phi)
                norm_psi = torch.mean(psi**2)
                norm_phi = torch.mean(phi**2)
                loss = loss + ortho_weight * overlap**2 / (norm_psi * norm_phi + 1e-12)

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


def pinn_eigenstates(V_func, a, b, n_states=2, **kw):
    models = []
    for _ in range(n_states):
        models.append(train_eigenstate(V_func, a, b, prev_states=models, **kw))
    return models


def evaluate_pinn(model, x_np):
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        psi = model(x_t).cpu().numpy().flatten()
    psi = psi / np.sqrt(np.trapz(psi**2, x_np))
    return psi


def pinn_energy(model, V_func, a, b, n=20000):
    x = torch.rand(n, 1, device=device) * (b - a) + a
    with torch.enable_grad():
        E = energy_functional(model, V_func, x).item()
    return E


# ===========================================================================
# 3. PROBLEM DEFINITIONS
# ===========================================================================
def run_box():
    L = 1.0
    a, b = 0.0, L
    V_func = lambda x: torch.zeros_like(x)
    V_np = lambda x: np.zeros_like(x)
    analytic_E = lambda n: (n * np.pi)**2 * hbar**2 / (2 * m * L**2)
    analytic_psi = lambda n, x: np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    return dict(name="Infinite Square Well", a=a, b=b, L=L,
                V_func=V_func, V_np=V_np,
                analytic_E=analytic_E, analytic_psi=analytic_psi,
                quantum_numbers=[1, 2])


def run_harmonic():
    omega = 1.0
    a, b = -6.0, 6.0
    V_func = lambda x: 0.5 * m * omega**2 * x**2
    V_np = lambda x: 0.5 * m * omega**2 * x**2

    from numpy.polynomial.hermite import hermval

    analytic_E = lambda n: hbar * omega * (n + 0.5)

    def analytic_psi(n, x):
        xi = np.sqrt(m * omega / hbar) * x
        coeff = np.zeros(n + 1)
        coeff[n] = 1.0
        Hn = hermval(xi, coeff)
        prefac = (m * omega / (np.pi * hbar))**0.25
        norm = 1.0 / np.sqrt(2**n * np.math.factorial(n))
        return prefac * norm * Hn * np.exp(-xi**2 / 2)

    return dict(name="Harmonic Oscillator", a=a, b=b, omega=omega,
                V_func=V_func, V_np=V_np,
                analytic_E=analytic_E, analytic_psi=analytic_psi,
                quantum_numbers=[0, 1])


def sign_align(psi, ref, x):
    if np.trapz(psi * ref, x) < 0:
        return -psi
    return psi


# ===========================================================================
# 4. MAIN
# ===========================================================================
def main(seed=0, save_plot=True):
    seed_everything(seed)
    problems = [run_box(), run_harmonic()]
    N_grid = 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) if save_plot else (None, None)
    summary_rows = []

    for row, P in enumerate(problems):
        a, b = P["a"], P["b"]
        x = np.linspace(a, b, N_grid)
        x_fd = x[1:-1]
        V_fd = P["V_np"](x_fd)

        print("\n" + "=" * 64)
        print(f"  {P['name']} (seed={seed})")
        print("=" * 64)

        t0 = time.time()
        E_fd, psi_fd = fd_eigenstates(x_fd, V_fd, n_states=2)
        t_fd = time.time() - t0
        print(f"Finite-difference diagonalisation: {t_fd*1000:.1f} ms")

        t0 = time.time()
        pinn_models = pinn_eigenstates(P["V_func"], a, b, n_states=2)
        t_pinn = time.time() - t0
        print(f"PINN training (2 states):          {t_pinn:.1f} s")

        colors = ["#1f77b4", "#d62728"]

        for k, n in enumerate(P["quantum_numbers"]):
            psi_exact = P["analytic_psi"](n, x)
            psi_exact = psi_exact / np.sqrt(np.trapz(psi_exact**2, x))
            E_exact = P["analytic_E"](n)

            psi_fd_k = sign_align(psi_fd[k], P["analytic_psi"](n, x_fd), x_fd)

            psi_pinn = evaluate_pinn(pinn_models[k], x)
            psi_pinn = sign_align(psi_pinn, psi_exact, x)
            E_pinn = pinn_energy(pinn_models[k], P["V_func"], a, b)

            l2_fd = np.sqrt(np.trapz((psi_fd_k - P["analytic_psi"](n, x_fd)
                              / np.sqrt(np.trapz(P["analytic_psi"](n, x_fd)**2, x_fd)))**2, x_fd)) * 100
            l2_pinn = np.sqrt(np.trapz((psi_pinn - psi_exact)**2, x)) * 100
            eE_fd = abs(E_fd[k] - E_exact) / E_exact * 100
            eE_pinn = abs(E_pinn - E_exact) / E_exact * 100

            print(f"\n  State n={n}:  E_analytic={E_exact:.5f}")
            print(f"    FD   : E={E_fd[k]:.5f} ({eE_fd:.3f}% err) | psi L2 err {l2_fd:.3f}%")
            print(f"    PINN : E={E_pinn:.5f} ({eE_pinn:.3f}% err) | psi L2 err {l2_pinn:.3f}%")

            summary_rows.append((P["name"], n, E_exact, E_fd[k], E_pinn, eE_pinn, l2_pinn))

            if save_plot:
                ax_target = axes[row][k]
                ax_target.plot(x, psi_exact, color="k", lw=3, alpha=0.4, label="Analytic (exact)")
                ax_target.plot(x_fd, psi_fd_k, color=colors[0], lw=1.8, ls=":",
                               label=f"Finite-diff ({eE_fd:.2g}% E-err)")
                ax_target.plot(x, psi_pinn, color=colors[1], lw=2, ls="--",
                               label=f"PINN ({eE_pinn:.2g}% E-err)")
                ax_target.set_title(f"{P['name']}: n={n}  (E={E_exact:.4f})",
                                    fontsize=12, fontweight="bold")
                ax_target.set_xlabel("x"); ax_target.set_ylabel("psi(x)")
                ax_target.legend(fontsize=9, loc="best")
                ax_target.grid(True, alpha=0.3)

    if save_plot:
        plt.suptitle(
            "Experiment 04: Zero-Momentum Eigenstates -- PINN matches the traditional "
            "solver\n(same machinery that FAILED on high-momentum dynamics)",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(FIGURES_DIR, "exp04_eigenstates.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")

    max_pinn_E_err = max(r[5] for r in summary_rows)
    max_pinn_l2 = max(r[6] for r in summary_rows)
    print("\n" + "=" * 64)
    print("  VERDICT")
    print("=" * 64)
    print(f"Worst PINN energy error:  {max_pinn_E_err:.3f}%")
    print(f"Worst PINN psi L2 error:  {max_pinn_l2:.3f}%")
    print("=" * 64)

    # --- Persist per-seed results for RESULTS.md aggregation (Task 2.6) ---
    result_path = os.path.join(RESULTS_DIR, "exp04_eigenstates.json")
    data = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
    data[str(seed)] = {
        "rows": [{"problem": r[0], "n": r[1], "E_exact": r[2], "E_fd": r[3],
                   "E_pinn": r[4], "E_pinn_err_pct": r[5], "psi_l2_err_pct": r[6]}
                  for r in summary_rows],
        "max_E_err_pct": max_pinn_E_err,
        "max_l2_err_pct": max_pinn_l2,
    }
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed, save_plot=(args.seed == 0))
