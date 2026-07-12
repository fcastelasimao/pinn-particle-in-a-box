"""
Shared 1D time-dependent Schrodinger PINN machinery.

Used by exp03 (original diagnosis) and the mechanism experiments added in
Phase 2 (exp06 momentum sweep, exp07 norm constraint, exp08 IC formulation,
exp09 causal training). Factored out on purpose: the IC-formulation
inconsistency between the 1D and 2D experiments (soft vs hard, found by the
project owner) was exactly the kind of bug that duplicated, drifting copies
of this code invites. One physics implementation, selected by keyword
arguments, removes that failure mode for the new experiments.

Physics: i*hbar d(psi)/dt = -hbar^2/2m d^2(psi)/dx^2, x in [-5,5], t in [0,1],
hard walls at the domain edges (spatial boundary condition, always hard,
enforced architecturally in Net1D — this is separate from the *initial*
condition in time, which is what 'ic=' below controls).
"""

import json
import os
import time

import numpy as np
from scipy.fft import dst, idst
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from qbox.utils import get_device, seed_everything

device = get_device()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_raw")


def get_adopted_ic(default="hard", default_w_ic=1000.0):
    """Read the IC formulation (and, if soft, the winning w_ic) decided by
    exp08_ic_formulation.py. Returns (ic, w_ic) -- w_ic is meaningless when
    ic == 'hard' but always returned for a uniform call signature.

    Falls back to defaults with a loud warning if exp08 hasn't been run yet
    -- exp06/exp07/exp09 should all be run AFTER exp08 (see IMPLEMENTATION_PLAN.md
    Phase 2 ordering note), but this keeps them runnable standalone too.
    """
    path = os.path.join(RESULTS_DIR, "exp08_ic_formulation.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- exp08 has not been run yet. "
              f"Falling back to ic='{default}'. Run exp08 first for the real decision.")
        return default, default_w_ic
    with open(path) as f:
        data = json.load(f)
    decision = data["decision"]
    w_ic = float(data.get("best_w_ic", default_w_ic))
    if decision == "soft":
        print(f"Adopted IC formulation (from exp08): soft, w_ic={w_ic}")
    else:
        print(f"Adopted IC formulation (from exp08): {decision}")
    return decision, w_ic

hbar = 1.0
m = 1.0
L = 10.0
x_lo = -L / 2
x_hi = L / 2
T = 1.0


def make_psi0(x0=0.0, sigma=1.0, px=1.0, norm_grid=1000):
    """Gaussian wave packet psi0(x) = A * exp(-(x-x0)^2/2sigma^2) * exp(i*px*x).

    Returns a callable psi0(x_np) -> (real, imag), normalised so
    integral |psi0|^2 dx = 1 (norm computed once on a dense reference grid).
    """
    x_dense = np.linspace(x_lo, x_hi, norm_grid)
    g_dense = np.exp(-((x_dense - x0) ** 2) / (2 * sigma ** 2))
    norm = np.sqrt(np.trapz(g_dense ** 2, x_dense))

    def psi0(x_np):
        gauss = np.exp(-((x_np - x0) ** 2) / (2 * sigma ** 2))
        phase_r = np.cos(px * x_np / hbar)
        phase_i = np.sin(px * x_np / hbar)
        return gauss * phase_r / norm, gauss * phase_i / norm

    return psi0


def split_operator_1d(psi0_func, N=512, n_steps=2000, frame_times=None):
    """1D ground truth for an arbitrary psi0_func: free particle in an
    infinite square well on [x_lo, x_hi], matching Net1D's architectural
    hard-wall BC (psi forced to exactly 0 at x_lo/x_hi, V=0 inside, no
    potential term in the PINN residual at all).

    Solved exactly via a Dirichlet sine series (DST-I): psi=0 at the walls
    is then exact by construction, eigenmodes evolve analytically
    (psi_n(t) = psi_n(0) exp(-i E_n t/hbar)), so there is no time-stepping
    error and no wall-height/stiffness hyperparameter to tune. (An earlier
    version used a finite V=50 soft wall 0.3 inside the domain edges as a
    split-operator FFT trick -- that let up to ~13% of the wavefunction's
    probability mass reach into the wall region for fast packets (px=4),
    diverging from this exact solution by >80% at t=1; low-momentum cases
    were only mildly affected, <2.3%. n_steps is accepted for call-site
    compatibility but unused.)
    """
    x = np.linspace(x_lo, x_hi, N + 2)[1:-1]  # interior points; psi=0 at the endpoints exactly
    dx = x[1] - x[0]
    Lbox = x_hi - x_lo
    n = np.arange(1, N + 1)
    E_n = hbar ** 2 * (n * np.pi / Lbox) ** 2 / (2 * m)

    psi_r, psi_i = psi0_func(x)
    psi = psi_r + 1j * psi_i
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi /= norm

    c_r = dst(psi.real, type=1)
    c_i = dst(psi.imag, type=1)

    if frame_times is None:
        frame_times = []
    frames = {}
    for ft in frame_times:
        phase = np.exp(-1j * E_n * ft / hbar)
        c = (c_r + 1j * c_i) * phase
        psi_t = idst(c.real, type=1) + 1j * idst(c.imag, type=1)
        frames[ft] = (x, np.abs(psi_t) ** 2, psi_t)

    return frames


class Net1D(nn.Module):
    """(x,t) -> (out_real, out_imag). Hard spatial-wall BC baked in via a
    multiplicative envelope that is exactly 0 at x_lo and x_hi."""

    def __init__(self, hidden_size=64, n_layers=5):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 2)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.bc_scale = (L / 2) ** 2

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        bc = (x - x_lo) * (x_hi - x) / self.bc_scale
        return bc * out[:, 0:1], bc * out[:, 1:2]


class PINNSolver1DTD:
    """
    1D time-dependent Schrodinger PINN with a selectable time-IC formulation.

    ic='hard' (default, matches the original exp03):
        psi(x,t) = psi0(x) + t * net(x,t)   -- IC satisfied exactly, no IC
        loss term, no weight hyperparameter.
    ic='soft':
        psi(x,t) = net(x,t)  (still hard-wall in space); IC enforced by
        loss = loss_pde + w_ic * loss_ic, loss_ic = mean squared error of
        net(x,0) against psi0(x).
    """

    def __init__(self, px=1.0, sigma=1.0, x0=0.0, seed=0, ic="hard",
                 w_ic=1000.0, hidden_size=64, n_layers=5):
        seed_everything(seed)
        self.px, self.sigma, self.x0 = px, sigma, x0
        self.ic = ic
        self.w_ic = w_ic
        self.psi0_func = make_psi0(x0, sigma, px)
        x_dense = np.linspace(x_lo, x_hi, 1000)
        g_dense = np.exp(-((x_dense - x0) ** 2) / (2 * sigma ** 2))
        self._psi0_norm = float(np.sqrt(np.trapz(g_dense ** 2, x_dense)))
        self.model = Net1D(hidden_size, n_layers).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.loss_history = []

    def psi0_at(self, x_tensor):
        x_np = x_tensor.detach().cpu().numpy().flatten()
        r, i = self.psi0_func(x_np)
        r_t = torch.tensor(r.reshape(-1, 1), dtype=torch.float32, device=device)
        i_t = torch.tensor(i.reshape(-1, 1), dtype=torch.float32, device=device)
        return r_t, i_t

    def _psi0_torch(self, x):
        """Differentiable psi0(x), matching make_psi0's normalisation. Used
        in full_psi (hard IC) so the PDE residual sees d(psi0)/dx via
        autograd -- psi0_at() detaches x to numpy and is only safe as a
        constant MSE target (ic_loss), never inside a term autograd needs
        to differentiate w.r.t. x."""
        gauss = torch.exp(-((x - self.x0) ** 2) / (2 * self.sigma ** 2))
        r = gauss * torch.cos(self.px * x / hbar) / self._psi0_norm
        i = gauss * torch.sin(self.px * x / hbar) / self._psi0_norm
        return r, i

    def full_psi(self, x, t):
        out_r, out_i = self.model(x, t)
        if self.ic == "hard":
            psi0_r, psi0_i = self._psi0_torch(x)
            return psi0_r + t * out_r, psi0_i + t * out_i
        return out_r, out_i

    def _pde_residual_terms(self, x, t):
        psi_r, psi_i = self.full_psi(x, t)

        psi_r_t = torch.autograd.grad(psi_r, t, torch.ones_like(psi_r), create_graph=True)[0]
        psi_i_t = torch.autograd.grad(psi_i, t, torch.ones_like(psi_i), create_graph=True)[0]

        psi_r_x = torch.autograd.grad(psi_r, x, torch.ones_like(psi_r), create_graph=True)[0]
        psi_r_xx = torch.autograd.grad(psi_r_x, x, torch.ones_like(psi_r_x), create_graph=True)[0]

        psi_i_x = torch.autograd.grad(psi_i, x, torch.ones_like(psi_i), create_graph=True)[0]
        psi_i_xx = torch.autograd.grad(psi_i_x, x, torch.ones_like(psi_i_x), create_graph=True)[0]

        res_r = -hbar * psi_i_t + (hbar ** 2 / (2 * m)) * psi_r_xx
        res_i = hbar * psi_r_t + (hbar ** 2 / (2 * m)) * psi_i_xx
        return res_r, res_i

    def per_point_residual(self, x, t):
        res_r, res_i = self._pde_residual_terms(x, t)
        return (res_r ** 2 + res_i ** 2).squeeze(-1)  # (N,), not reduced

    def pde_residual(self, x, t):
        res_r, res_i = self._pde_residual_terms(x, t)
        return torch.mean(res_r ** 2 + res_i ** 2)

    def ic_loss(self, n_points=500):
        x = torch.rand(n_points, 1, device=device) * L + x_lo
        t = torch.zeros_like(x)
        out_r, out_i = self.model(x, t)
        psi0_r, psi0_i = self.psi0_at(x)
        return torch.mean((out_r - psi0_r) ** 2 + (out_i - psi0_i) ** 2)

    def norm_loss(self, n_times=8, n_grid=256):
        """mean_t (N(t) - 1)^2, N(t) = integral |psi(x,t)|^2 dx (trapezoid)."""
        t_samples = torch.rand(n_times, device=device) * T
        x_grid = torch.linspace(x_lo, x_hi, n_grid, device=device).reshape(-1, 1)
        total = 0.0
        for tv in t_samples:
            t_col = torch.full_like(x_grid, float(tv))
            psi_r, psi_i = self.full_psi(x_grid, t_col)
            prob = (psi_r ** 2 + psi_i ** 2).squeeze(-1)
            N_t = torch.trapezoid(prob, x_grid.squeeze(-1))
            total = total + (N_t - 1.0) ** 2
        return total / n_times

    def train(self, n_collocation=2000, epochs=5000, norm_weight=0.0,
              causal=False, causal_bins=10, causal_eps=10.0, verbose=True):
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        start = time.time()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            x = torch.rand(n_collocation, 1, device=device) * L + x_lo
            t = torch.rand(n_collocation, 1, device=device) * T
            x.requires_grad_(True)
            t.requires_grad_(True)

            if causal:
                res2 = self.per_point_residual(x, t)
                bin_idx = torch.clamp((t.detach().squeeze(-1) / T * causal_bins).long(),
                                       0, causal_bins - 1)
                bin_sums = torch.zeros(causal_bins, device=device)
                bin_counts = torch.zeros(causal_bins, device=device)
                bin_sums.scatter_add_(0, bin_idx, res2.detach())
                bin_counts.scatter_add_(0, bin_idx, torch.ones_like(res2))
                bin_means = bin_sums / bin_counts.clamp(min=1)
                cum_prior = torch.cumsum(bin_means, dim=0) - bin_means
                weights = torch.exp(-causal_eps * cum_prior)  # stop-gradient: built from .detach()'d data
                point_weights = weights[bin_idx]
                pde_loss = (point_weights * res2).sum() / point_weights.sum()
            else:
                pde_loss = self.pde_residual(x, t)

            loss = pde_loss
            if self.ic == "soft":
                loss = loss + self.w_ic * self.ic_loss()
            if norm_weight > 0:
                loss = loss + norm_weight * self.norm_loss()

            loss.backward()
            self.optimizer.step()
            scheduler.step()
            self.loss_history.append(loss.item())

            if verbose and epoch % 1000 == 0:
                print(f"    epoch {epoch:5d}/{epochs} | loss {loss.item():.4e}")

        if verbose:
            print(f"    done in {time.time() - start:.1f}s")

    def predict(self, x_np, t_val):
        self.model.eval()
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.ones_like(x_t) * t_val
        with torch.no_grad():
            psi_r, psi_i = self.full_psi(x_t, t_t)
            prob = (psi_r ** 2 + psi_i ** 2).cpu().numpy().flatten()
        self.model.train()
        return prob

    def norm_curve(self, t_vals, n_grid=256):
        """N(t) at explicit times, for plotting (no grad needed)."""
        x_grid = np.linspace(x_lo, x_hi, n_grid)
        out = []
        for tv in t_vals:
            prob = self.predict(x_grid, tv)
            out.append(np.trapz(prob, x_grid))
        return np.array(out)


def mean_relative_error(solver, gt_frames, frame_times, x_plot):
    """Mean relative L2 error of |psi|^2 across frame_times, matching exp03's metric."""
    errors = {}
    for t_val in frame_times:
        if t_val not in gt_frames:
            continue
        prob_pinn = solver.predict(x_plot, t_val)
        _, prob_gt, _ = gt_frames[t_val]
        prob_gt_interp = np.interp(x_plot, gt_frames[t_val][0], prob_gt)
        l2 = np.sqrt(np.trapz((prob_pinn - prob_gt_interp) ** 2, x_plot))
        rel = l2 / np.sqrt(np.trapz(prob_gt_interp ** 2, x_plot)) * 100
        errors[t_val] = rel
    return errors
