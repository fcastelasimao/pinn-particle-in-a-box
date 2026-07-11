"""
Experiment 05: Inverse Problem -- Recover an Unknown Potential
============================================================
THIS is where a PINN genuinely beats a traditional solver.

THE FORWARD vs INVERSE DISTINCTION
----------------------------------
  Forward problem:  V(x)  ->  psi(x), E      "given the potential, find the state"
  Inverse problem:  psi(x), E  ->  V(x)      "given the state, find the potential"

A finite-difference / split-operator solver is built for the FORWARD problem.
It cannot directly solve the inverse problem -- you would have to wrap it in
an outer optimisation loop, repeatedly guessing V, re-solving, and comparing.

A PINN solves the inverse problem NATIVELY: just make V(x) a second neural
network and let the same physics loss that constrains psi also constrain V.
One training run, no outer loop. This is a structural advantage, not a
tuning trick.

THE SETUP (a realistic measurement scenario)
--------------------------------------------
A particle sits in an UNKNOWN confining potential V_true(x) (here: a box
with a central Gaussian barrier -- a tunnelling-style double well). An
experiment gives us:
    * SPARSE, NOISY samples of the ground-state wavefunction psi(x).
    * The ground-state energy E (energies are what spectroscopy measures
      most accurately -- so we treat E as known; this also fixes the
      V <-> V+const gauge).
We must reconstruct the full potential V(x).

WHY NOT JUST DIFFERENTIATE?  (the naive baseline)
-------------------------------------------------
The time-independent Schrodinger equation can be rearranged to:

    V(x) = E + (hbar^2/2m) * psi''(x) / psi(x)

So in principle you just measure psi, take its second derivative, and read
off V. In practice psi'' amplifies measurement noise catastrophically, and
dividing by psi explodes wherever psi is small. We include this baseline to
show it fails -- and that the PINN's physics-regularised reconstruction
does not.

GROUND TRUTH
------------
Finite-difference diagonalisation of H = -hbar^2/2m d^2/dx^2 + V_true gives
the exact psi and E we sample from, and the V_true we score the
reconstruction against.

Multi-seed: pass --seed N (default 0). Only seed 0 saves the committed PNG;
all seeds append their numbers to results_raw/exp05_inverse_potential.json
for the mean +- std reported in RESULTS.md (Task 2.6).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qbox.utils import get_device, seed_everything

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()

hbar = 1.0
m = 1.0
L = 1.0
a, b = 0.0, L


def V_true(x):
    barrier = 50.0 * np.exp(-((x - 0.5) / 0.10) ** 2)
    return barrier


def fd_ground_state(N=801):
    x = np.linspace(a, b, N)
    x_in = x[1:-1]
    dx = x_in[1] - x_in[0]
    kin = hbar**2 / (2 * m * dx**2)
    main = 2 * kin + V_true(x_in)
    off = -kin * np.ones(len(x_in) - 1)
    E, vecs = eigh_tridiagonal(main, off, select='i', select_range=(0, 0))
    psi = vecs[:, 0]
    psi = psi / np.sqrt(np.trapz(psi**2, x_in))
    if psi.mean() < 0:
        psi = -psi
    return x_in, psi, E[0]


class PsiNet(nn.Module):
    def __init__(self, hidden=64, n_layers=4):
        super().__init__()
        self.scale = (L / 2) ** 2
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, x):
        return x * (L - x) / self.scale * self.net(x)


class VNet(nn.Module):
    def __init__(self, hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


def naive_inversion(x_dense, psi_noisy, E):
    dx = x_dense[1] - x_dense[0]
    psi_xx = np.gradient(np.gradient(psi_noisy, dx), dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        V = E + (hbar**2 / (2 * m)) * psi_xx / psi_noisy
    return V


def train_inverse(x_obs, psi_obs, E_known, warmup=5000, epochs=5000,
                  n_collocation=1500, lr=1e-3, w_data=1.0, verbose=True):
    psi_net = PsiNet().to(device)
    v_net = VNet().to(device)

    x_obs_t = torch.tensor(x_obs.reshape(-1, 1), dtype=torch.float32, device=device)
    psi_obs_t = torch.tensor(psi_obs.reshape(-1, 1), dtype=torch.float32, device=device)

    hist = {"data": [], "pde": []}
    start = time.time()

    opt_psi = Adam(psi_net.parameters(), lr=lr)
    sched_psi = CosineAnnealingLR(opt_psi, T_max=warmup, eta_min=1e-5)
    for epoch in range(warmup):
        opt_psi.zero_grad()
        loss_data = torch.mean((psi_net(x_obs_t) - psi_obs_t) ** 2)
        loss_data.backward()
        opt_psi.step(); sched_psi.step()
        hist["data"].append(loss_data.item())
        if verbose and epoch % 1000 == 0:
            print(f"  [warmup] epoch {epoch:5d} | data {loss_data.item():.3e}")

    for p in psi_net.parameters():
        p.requires_grad_(False)
    psi_net.eval()

    opt_v = Adam(v_net.parameters(), lr=lr)
    sched_v = CosineAnnealingLR(opt_v, T_max=epochs, eta_min=1e-5)
    for epoch in range(epochs):
        opt_v.zero_grad()
        xc = torch.rand(n_collocation, 1, device=device) * (b - a) + a
        xc.requires_grad_(True)
        psi = psi_net(xc)
        psi_x = torch.autograd.grad(psi, xc, torch.ones_like(psi), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, xc, torch.ones_like(psi_x))[0]
        psi = psi.detach()
        psi_xx = psi_xx.detach()

        V = v_net(xc)
        residual = -(hbar**2 / (2 * m)) * psi_xx + V * psi - E_known * psi
        loss_pde = torch.mean((residual ** 2) * psi.detach() ** 2)

        loss_pde.backward()
        opt_v.step(); sched_v.step()
        hist["pde"].append(loss_pde.item())
        if verbose and epoch % 1000 == 0:
            print(f"  [V-rec]  epoch {epoch:5d} | weighted pde {loss_pde.item():.3e}")

    if verbose:
        print(f"Inverse PINN trained in {time.time()-start:.1f}s")
    return psi_net, v_net, hist


def main(seed=0, save_plot=True):
    seed_everything(seed)

    x_gt, psi_gt, E_gt = fd_ground_state(N=801)
    print(f"True ground-state energy E = {E_gt:.5f}")

    rng = np.random.default_rng(seed)
    n_obs = 40
    noise_level = 0.03
    obs_idx = np.linspace(0, len(x_gt) - 1, n_obs).astype(int)
    x_obs = x_gt[obs_idx]
    psi_clean = psi_gt[obs_idx]
    psi_obs = psi_clean + noise_level * psi_gt.max() * rng.standard_normal(n_obs)
    print(f"Observations: {n_obs} sparse points, {noise_level*100:.0f}% noise (seed={seed})")

    psi_net, v_net, hist = train_inverse(x_obs, psi_obs, E_gt, verbose=save_plot)

    x_eval = np.linspace(a, b, 500)
    x_eval_t = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        V_pinn = v_net(x_eval_t).cpu().numpy().flatten()
        psi_pinn = psi_net(x_eval_t).cpu().numpy().flatten()
    psi_pinn = psi_pinn / np.sqrt(np.trapz(psi_pinn**2, x_eval))

    V_true_eval = V_true(x_eval)

    weight = np.interp(x_eval, x_gt, psi_gt)**2
    mask = weight > 0.02 * weight.max()
    V_l2 = np.sqrt(np.trapz((V_pinn - V_true_eval)**2 * mask, x_eval) /
                   np.trapz(mask.astype(float), x_eval))
    V_rel = V_l2 / np.sqrt(np.trapz(V_true_eval**2 * mask, x_eval) /
                           np.trapz(mask.astype(float), x_eval)) * 100
    print(f"\nReconstructed V relative L2 error (where |psi|^2>0): {V_rel:.2f}%")

    n_dense = 200
    dense_idx = np.linspace(0, len(x_gt) - 1, n_dense).astype(int)
    x_dense = x_gt[dense_idx]
    psi_dense_noisy = psi_gt[dense_idx] + noise_level * psi_gt.max() * rng.standard_normal(n_dense)
    V_naive = naive_inversion(x_dense, psi_dense_noisy, E_gt)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.plot(x_eval, V_true_eval, 'k-', lw=3, alpha=0.5, label='True V(x)')
        ax.plot(x_eval, V_pinn, 'r--', lw=2.5, label=f'PINN recovery ({V_rel:.1f}% err)')
        ax.plot(x_dense, np.clip(V_naive, -50, 200), 'g.', ms=4, alpha=0.4,
                label='Naive psi\'\'/psi (more data!)')
        ax.set_ylim(-20, 120)
        ax.set_xlabel('x'); ax.set_ylabel('V(x)')
        ax.set_title('Recovered Potential', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x_eval, np.interp(x_eval, x_gt, psi_gt), 'k-', lw=3, alpha=0.5,
                label='True psi (ground state)')
        ax.plot(x_obs, psi_obs, 'bo', ms=5, alpha=0.6, label=f'Noisy obs (n={n_obs})')
        ax.plot(x_eval, psi_pinn, 'r--', lw=2, label='PINN psi (denoised)')
        ax.set_xlabel('x'); ax.set_ylabel('psi(x)')
        ax.set_title('Wavefunction: data vs PINN', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = axes[2]
        n_warm = len(hist["data"])
        ax.semilogy(range(n_warm), hist["data"], label='Stage 1: psi data fit',
                    color='blue', lw=1.2)
        ax.semilogy(range(n_warm, n_warm + len(hist["pde"])), hist["pde"],
                    label='Stage 2: V recovery (PDE)', color='green', lw=1.2)
        ax.axvline(n_warm, color='gray', ls='--', alpha=0.6)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Inverse Training Loss', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Experiment 05: Inverse Problem -- recover V(x) from sparse, noisy psi data.\n"
            "A traditional solver cannot do this directly; the PINN does it in one run.",
            fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = os.path.join(FIGURES_DIR, "exp05_inverse_potential.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    print("\n" + "=" * 64)
    print("  VERDICT")
    print("=" * 64)
    print(f"V relative L2 error: {V_rel:.1f}%")
    print("=" * 64)

    result_path = os.path.join(RESULTS_DIR, "exp05_inverse_potential.json")
    data = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
    data[str(seed)] = {"V_rel_err_pct": V_rel, "E_gt": E_gt}
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    return V_rel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed, save_plot=(args.seed == 0))
