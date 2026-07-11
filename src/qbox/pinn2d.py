"""
PINN (Physics-Informed Neural Network) Solver for 2D Schrödinger Equation
==========================================================================

This solver uses a neural network to learn the wave function ψ(x,y,t) by
minimizing a loss function that encodes the Schrödinger equation itself.

Key idea: No training data needed! The physics IS the training signal.

Two IC formulations are provided:
  - PINNSolver       : SOFT initial condition (loss_pde + 1000 * loss_ic).
                        Original formulation.
  - PINNSolverHardIC : HARD initial condition (psi = psi0 + t*net), added in
                        Phase 2 (Task 2.2) so the 2D headline result can be
                        re-verified against the formulation adopted in
                        exp08_ic_formulation.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
import torch
import torch.nn as nn
from torch.optim import Adam

from qbox.utils import get_device

device = get_device()


class SchrodingerPINN(nn.Module):
    """
    Physics-Informed Neural Network for the 2D Schrödinger equation.

    Network architecture: Takes (x, y, t) as input, outputs real and imaginary
    parts of the wave function ψ(x,y,t) = ψ_real + i*ψ_imag
    """

    def __init__(self, layers=[3, 128, 128, 128, 128, 2]):
        """
        Initialize the neural network.

        Parameters:
        -----------
        layers : list
            Network architecture. First element is input dimension (3 for x,y,t),
            last element is output dimension (2 for real and imaginary parts),
            intermediate elements are hidden layer sizes.
        """
        super(SchrodingerPINN, self).__init__()

        # Build the network
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))

        self.activation = nn.Tanh()  # Tanh works well for smooth functions

        # Initialize weights using Xavier initialization
        for layer in self.layers_list:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x, y, t : torch tensors
            Coordinates (can be batches)

        Returns:
        --------
        psi_real, psi_imag : torch tensors
            Real and imaginary parts of ψ(x,y,t)
        """
        # Concatenate inputs
        inputs = torch.cat([x, y, t], dim=1)

        # Pass through hidden layers
        out = inputs
        for i, layer in enumerate(self.layers_list[:-1]):
            out = self.activation(layer(out))

        # Output layer (no activation)
        out = self.layers_list[-1](out)

        # Split into real and imaginary parts
        psi_real = out[:, 0:1]
        psi_imag = out[:, 1:2]

        return psi_real, psi_imag


class PINNSolver:
    """
    PINN solver for the 2D time-dependent Schrödinger equation.
    SOFT initial condition: loss = loss_pde + 1000.0 * loss_initial.
    """

    def __init__(self, X, Y, V, hbar=1.0, m=1.0):
        """
        Initialize the PINN solver.

        Parameters:
        -----------
        X, Y : 2D numpy arrays
            Spatial grid
        V : 2D numpy array
            Potential landscape
        hbar, m : float
            Physical constants
        """
        self.X = X
        self.Y = Y
        self.V = V
        self.hbar = hbar
        self.m = m

        # Extract grid properties
        self.N = X.shape[0]
        self.L = X.max() - X.min()

        # Create the neural network
        self.model = SchrodingerPINN(layers=[3, 128, 128, 128, 128, 2]).to(device)

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        # Loss history
        self.loss_history = []

    def potential(self, x, y):
        """
        Get potential value at given coordinates using interpolation.

        Parameters:
        -----------
        x, y : torch tensors
            Coordinates

        Returns:
        --------
        V : torch tensor
            Potential values
        """
        # Convert to numpy for interpolation
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()

        # Simple nearest-neighbor interpolation
        # (Could use scipy.interpolate for better accuracy)
        x_idx = np.clip(((x_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)
        y_idx = np.clip(((y_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)

        V_vals = self.V[y_idx, x_idx]

        return torch.tensor(V_vals, dtype=torch.float32, device=device).reshape(-1, 1)

    def compute_pde_residual(self, x, y, t):
        """
        Compute the residual of the Schrödinger equation.

        The Schrödinger equation is:
        iℏ ∂ψ/∂t = -ℏ²/(2m) ∇²ψ + V ψ

        Rearranged: iℏ ∂ψ/∂t + ℏ²/(2m) ∇²ψ - V ψ = 0

        We want this residual to be as close to zero as possible.

        Parameters:
        -----------
        x, y, t : torch tensors with requires_grad=True
            Coordinates

        Returns:
        --------
        residual : torch tensor
            PDE residual (complex)
        """
        # Get wave function
        psi_real, psi_imag = self.model(x, y, t)

        # Compute derivatives using autograd
        # Time derivatives
        psi_real_t = torch.autograd.grad(psi_real, t,
                                         grad_outputs=torch.ones_like(psi_real),
                                         create_graph=True)[0]
        psi_imag_t = torch.autograd.grad(psi_imag, t,
                                         grad_outputs=torch.ones_like(psi_imag),
                                         create_graph=True)[0]

        # Spatial derivatives for real part
        psi_real_x = torch.autograd.grad(psi_real, x,
                                         grad_outputs=torch.ones_like(psi_real),
                                         create_graph=True)[0]
        psi_real_xx = torch.autograd.grad(psi_real_x, x,
                                          grad_outputs=torch.ones_like(psi_real_x),
                                          create_graph=True)[0]
        psi_real_y = torch.autograd.grad(psi_real, y,
                                         grad_outputs=torch.ones_like(psi_real),
                                         create_graph=True)[0]
        psi_real_yy = torch.autograd.grad(psi_real_y, y,
                                          grad_outputs=torch.ones_like(psi_real_y),
                                          create_graph=True)[0]

        # Spatial derivatives for imaginary part
        psi_imag_x = torch.autograd.grad(psi_imag, x,
                                         grad_outputs=torch.ones_like(psi_imag),
                                         create_graph=True)[0]
        psi_imag_xx = torch.autograd.grad(psi_imag_x, x,
                                          grad_outputs=torch.ones_like(psi_imag_x),
                                          create_graph=True)[0]
        psi_imag_y = torch.autograd.grad(psi_imag, y,
                                         grad_outputs=torch.ones_like(psi_imag),
                                         create_graph=True)[0]
        psi_imag_yy = torch.autograd.grad(psi_imag_y, y,
                                          grad_outputs=torch.ones_like(psi_imag_y),
                                          create_graph=True)[0]

        # Laplacian
        laplacian_real = psi_real_xx + psi_real_yy
        laplacian_imag = psi_imag_xx + psi_imag_yy

        # Get potential
        V = self.potential(x, y)

        # Schrödinger equation: iℏ ∂ψ/∂t = -ℏ²/(2m) ∇²ψ + V ψ
        # Separating real and imaginary parts:
        # Real part: -ℏ ∂(Im ψ)/∂t = -ℏ²/(2m) ∇²(Re ψ) + V (Re ψ)
        # Imag part:  ℏ ∂(Re ψ)/∂t = -ℏ²/(2m) ∇²(Im ψ) + V (Im ψ)

        residual_real = -self.hbar * psi_imag_t + self.hbar**2 / (2*self.m) * laplacian_real - V * psi_real
        residual_imag = self.hbar * psi_real_t + self.hbar**2 / (2*self.m) * laplacian_imag - V * psi_imag

        return residual_real, residual_imag

    def loss_function(self, x_collocation, y_collocation, t_collocation,
                     x_initial, y_initial, psi_initial_real, psi_initial_imag):
        """
        Compute the total loss: PDE residual + initial condition.

        Parameters:
        -----------
        x_collocation, y_collocation, t_collocation : torch tensors
            Points where we enforce the PDE (collocation points)
        x_initial, y_initial : torch tensors
            Points where we enforce initial condition (t=0)
        psi_initial_real, psi_initial_imag : torch tensors
            Initial wave function values

        Returns:
        --------
        loss : torch tensor
            Total loss
        """
        # PDE residual loss
        residual_real, residual_imag = self.compute_pde_residual(
            x_collocation, y_collocation, t_collocation
        )
        loss_pde = torch.mean(residual_real**2 + residual_imag**2)

        # Initial condition loss
        psi_real_pred, psi_imag_pred = self.model(x_initial, y_initial,
                                                   torch.zeros_like(x_initial))
        loss_initial = torch.mean((psi_real_pred - psi_initial_real)**2 +
                                 (psi_imag_pred - psi_initial_imag)**2)

        # Total loss (weight initial condition more heavily)
        total_loss = loss_pde + 1000.0 * loss_initial

        return total_loss, loss_pde, loss_initial

    def train(self, psi_initial, t_max=1.0, n_collocation=2000,
             n_initial=2000, epochs=5000):
        """
        Train the PINN.

        Parameters:
        -----------
        psi_initial : 2D complex numpy array
            Initial wave function at t=0
        t_max : float
            Maximum time to train for
        n_collocation : int
            Number of random collocation points for PDE
        n_initial : int
            Number of points for initial condition
        epochs : int
            Number of training epochs
        """
        print(f"\nTraining PINN for {epochs} epochs...")
        print(f"Device: {device}")

        # Prepare initial condition data
        x_flat = self.X.flatten()
        y_flat = self.Y.flatten()
        psi_real_flat = np.real(psi_initial).flatten()
        psi_imag_flat = np.imag(psi_initial).flatten()

        # Sample random points for initial condition
        idx_initial = np.random.choice(len(x_flat), n_initial, replace=False)
        x_initial = torch.tensor(x_flat[idx_initial], dtype=torch.float32, device=device).reshape(-1, 1)
        y_initial = torch.tensor(y_flat[idx_initial], dtype=torch.float32, device=device).reshape(-1, 1)
        psi_initial_real = torch.tensor(psi_real_flat[idx_initial], dtype=torch.float32, device=device).reshape(-1, 1)
        psi_initial_imag = torch.tensor(psi_imag_flat[idx_initial], dtype=torch.float32, device=device).reshape(-1, 1)

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Sample random collocation points in space-time
            x_coll = torch.rand(n_collocation, 1, device=device) * self.L - self.L/2
            y_coll = torch.rand(n_collocation, 1, device=device) * self.L - self.L/2
            t_coll = torch.rand(n_collocation, 1, device=device) * t_max

            # Require gradients for computing derivatives
            x_coll.requires_grad = True
            y_coll.requires_grad = True
            t_coll.requires_grad = True

            # Compute loss
            loss, loss_pde, loss_ic = self.loss_function(
                x_coll, y_coll, t_coll,
                x_initial, y_initial, psi_initial_real, psi_initial_imag
            )

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Record loss
            self.loss_history.append(loss.item())

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f}")

        print("Training complete!")

    def predict(self, times):
        """
        Predict wave function at given times.

        Parameters:
        -----------
        times : list or array
            Times at which to predict

        Returns:
        --------
        frames : list of 2D arrays
            Probability density |ψ|² at each time
        """
        self.model.eval()  # Set to evaluation mode

        frames = []

        with torch.no_grad():
            for t in times:
                # Create grid for this time
                x_grid = torch.tensor(self.X.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
                y_grid = torch.tensor(self.Y.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
                t_grid = torch.ones_like(x_grid) * t

                # Predict
                psi_real, psi_imag = self.model(x_grid, y_grid, t_grid)

                # Convert to numpy and reshape
                psi_real_np = psi_real.cpu().numpy().reshape(self.X.shape)
                psi_imag_np = psi_imag.cpu().numpy().reshape(self.X.shape)

                # Compute probability density
                probability = psi_real_np**2 + psi_imag_np**2
                frames.append(probability)

        return frames

    def plot_loss(self, save_path='pinn_loss.png'):
        """Plot the training loss history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PINN Training Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Loss plot saved as '{save_path}'")


class Net2DDelta(nn.Module):
    """
    Maps (x, y, t) -> (delta_real, delta_imag): the deviation from the
    initial state, for the HARD-IC 2D formulation. Same construction as
    Net1D in exp03_td_1d_lowmom.py, one dimension up.
    """

    def __init__(self, hidden_size=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 2)]
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2]


class PINNSolverHardIC:
    """
    PINN solver for the 2D time-dependent Schrödinger equation.
    HARD initial condition: psi(x,y,t) = psi0(x,y) + t * net(x,y,t).
    No IC loss term, no IC weight — added in Task 2.2 to re-verify the 2D
    headline result against the formulation adopted in exp08_ic_formulation.py.
    """

    def __init__(self, X, Y, V, hbar=1.0, m=1.0):
        self.X = X
        self.Y = Y
        self.V = V
        self.hbar = hbar
        self.m = m

        self.N = X.shape[0]
        self.L = X.max() - X.min()

        self.model = Net2DDelta(hidden_size=128, n_layers=4).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.loss_history = []

        self.psi0_real = None
        self.psi0_imag = None

    def potential(self, x, y):
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        x_idx = np.clip(((x_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)
        y_idx = np.clip(((y_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)
        V_vals = self.V[y_idx, x_idx]
        return torch.tensor(V_vals, dtype=torch.float32, device=device).reshape(-1, 1)

    def psi0_at(self, x, y):
        """Nearest-grid-point lookup of psi0 (same interpolation style as potential())."""
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        x_idx = np.clip(((x_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)
        y_idx = np.clip(((y_np + self.L/2) / self.L * (self.N-1)).astype(int), 0, self.N-1)
        r_vals = self.psi0_real[y_idx, x_idx]
        i_vals = self.psi0_imag[y_idx, x_idx]
        r_t = torch.tensor(r_vals, dtype=torch.float32, device=device).reshape(-1, 1)
        i_t = torch.tensor(i_vals, dtype=torch.float32, device=device).reshape(-1, 1)
        return r_t, i_t

    def full_psi(self, x, y, t):
        psi0_r, psi0_i = self.psi0_at(x, y)
        delta_r, delta_i = self.model(x, y, t)
        psi_r = psi0_r + t * delta_r
        psi_i = psi0_i + t * delta_i
        return psi_r, psi_i

    def compute_pde_residual(self, x, y, t):
        psi_real, psi_imag = self.full_psi(x, y, t)

        psi_real_t = torch.autograd.grad(psi_real, t,
            grad_outputs=torch.ones_like(psi_real), create_graph=True)[0]
        psi_imag_t = torch.autograd.grad(psi_imag, t,
            grad_outputs=torch.ones_like(psi_imag), create_graph=True)[0]

        psi_real_x = torch.autograd.grad(psi_real, x,
            grad_outputs=torch.ones_like(psi_real), create_graph=True)[0]
        psi_real_xx = torch.autograd.grad(psi_real_x, x,
            grad_outputs=torch.ones_like(psi_real_x), create_graph=True)[0]
        psi_real_y = torch.autograd.grad(psi_real, y,
            grad_outputs=torch.ones_like(psi_real), create_graph=True)[0]
        psi_real_yy = torch.autograd.grad(psi_real_y, y,
            grad_outputs=torch.ones_like(psi_real_y), create_graph=True)[0]

        psi_imag_x = torch.autograd.grad(psi_imag, x,
            grad_outputs=torch.ones_like(psi_imag), create_graph=True)[0]
        psi_imag_xx = torch.autograd.grad(psi_imag_x, x,
            grad_outputs=torch.ones_like(psi_imag_x), create_graph=True)[0]
        psi_imag_y = torch.autograd.grad(psi_imag, y,
            grad_outputs=torch.ones_like(psi_imag), create_graph=True)[0]
        psi_imag_yy = torch.autograd.grad(psi_imag_y, y,
            grad_outputs=torch.ones_like(psi_imag_y), create_graph=True)[0]

        laplacian_real = psi_real_xx + psi_real_yy
        laplacian_imag = psi_imag_xx + psi_imag_yy

        V = self.potential(x, y)

        residual_real = -self.hbar * psi_imag_t + self.hbar**2 / (2*self.m) * laplacian_real - V * psi_real
        residual_imag = self.hbar * psi_real_t + self.hbar**2 / (2*self.m) * laplacian_imag - V * psi_imag

        return residual_real, residual_imag

    def train(self, psi_initial, t_max=1.0, n_collocation=2000, epochs=5000):
        print(f"\nTraining hard-IC PINN for {epochs} epochs...")
        print(f"Device: {device}")

        self.psi0_real = np.real(psi_initial)
        self.psi0_imag = np.imag(psi_initial)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            x_coll = torch.rand(n_collocation, 1, device=device) * self.L - self.L/2
            y_coll = torch.rand(n_collocation, 1, device=device) * self.L - self.L/2
            t_coll = torch.rand(n_collocation, 1, device=device) * t_max

            x_coll.requires_grad = True
            y_coll.requires_grad = True
            t_coll.requires_grad = True

            res_r, res_i = self.compute_pde_residual(x_coll, y_coll, t_coll)
            loss = torch.mean(res_r**2 + res_i**2)

            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | PDE loss: {loss.item():.6f}")

        print("Training complete!")

    def predict(self, times):
        self.model.eval()
        frames = []
        with torch.no_grad():
            for t in times:
                x_grid = torch.tensor(self.X.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
                y_grid = torch.tensor(self.Y.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
                t_grid = torch.ones_like(x_grid) * t

                psi_real, psi_imag = self.full_psi(x_grid, y_grid, t_grid)

                psi_real_np = psi_real.cpu().numpy().reshape(self.X.shape)
                psi_imag_np = psi_imag.cpu().numpy().reshape(self.X.shape)

                probability = psi_real_np**2 + psi_imag_np**2
                frames.append(probability)

        return frames
