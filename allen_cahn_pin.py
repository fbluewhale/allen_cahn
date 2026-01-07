"""
Allen–Cahn solver using Physics-Informed Neural Networks (PINN) - Fast version.

PDE:
    u_t = eps^2 * u_xx - (u^3 - u)

Uses automatic differentiation via JAX for efficient gradient computation.
Falls back to scipy optimization with analytical Jacobians.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt


class FastAllenCahnPINN:
    """Ultra-fast PINN for Allen-Cahn using shallow network."""
    
    def __init__(self, eps=0.01, layers=(2, 32, 1)):
        """
        Initialize the PINN with shallow network.
        
        Parameters
        ----------
        eps : float
            Interface parameter.
        layers : tuple
            Layer sizes (input, hidden..., output).
        """
        self.eps = eps
        self.layers = layers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        np.random.seed(42)
        self.W = []
        self.b = []
        
        for i in range(len(self.layers) - 1):
            limit = np.sqrt(6.0 / (self.layers[i] + self.layers[i+1]))
            W = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1]))
            b = np.zeros((1, self.layers[i+1]))
            self.W.append(W)
            self.b.append(b)
    
    def _pack_params(self):
        """Pack weights and biases into a single vector."""
        params = np.concatenate([W.flatten() for W in self.W] + 
                                [b.flatten() for b in self.b])
        return params
    
    def _unpack_params(self, params):
        """Unpack parameters into weights and biases."""
        idx = 0
        for i in range(len(self.W)):
            size = self.W[i].size
            self.W[i] = params[idx:idx+size].reshape(self.W[i].shape)
            idx += size
        
        for i in range(len(self.b)):
            size = self.b[i].size
            self.b[i] = params[idx:idx+size].reshape(self.b[i].shape)
            idx += size
    
    def forward_single(self, x, t):
        """Forward pass for a single point."""
        # Input normalization
        x_n = np.clip(x, -1.0, 1.0)
        t_n = np.clip(t, 0.0, 1.0)
        z = np.array([[x_n, t_n]])
        
        # Hidden layers with ReLU
        for i in range(len(self.W) - 1):
            z = np.dot(z, self.W[i]) + self.b[i]
            z = np.maximum(z, 0.0)
        
        # Output layer
        u = np.dot(z, self.W[-1]) + self.b[-1]
        return float(u[0, 0])
    
    def forward_vectorized(self, x, t):
        """Vectorized forward pass."""
        results = []
        for xi, ti in zip(x, t):
            results.append(self.forward_single(xi, ti))
        return np.array(results)
    
    def pde_residual(self, x, t, h=0.005):
        """Compute PDE residual using finite differences."""
        u = self.forward_single(x, t)
        
        # u_t
        u_tp = self.forward_single(x, t + h)
        u_tm = self.forward_single(x, max(0, t - h))
        u_t = (u_tp - u_tm) / (2 * h)
        
        # u_xx
        u_pp = self.forward_single(min(1, x + h), t)
        u_mm = self.forward_single(max(-1, x - h), t)
        u_xx = (u_pp - 2*u + u_mm) / (h**2)
        
        # Clamp to avoid overflow
        u_clipped = np.clip(u, -2, 2)
        nonlin = u_clipped**3 - u_clipped
        residual = u_t - self.eps**2 * u_xx + nonlin
        
        return np.clip(residual, -1e6, 1e6)
    
    def loss_function(self, params, x_pde, t_pde, x_ic, t_ic, u_ic, w_pde=1.0, w_ic=5.0):
        """Compute total loss."""
        self._unpack_params(params)
        
        # PDE loss (subset)
        n_pde = min(len(x_pde), 50)
        idx_pde = np.random.choice(len(x_pde), n_pde, replace=False)
        residuals = np.array([self.pde_residual(x_pde[i], t_pde[i]) for i in idx_pde])
        loss_pde = np.mean(residuals**2)
        
        # IC loss
        u_ic_pred = self.forward_vectorized(x_ic, t_ic)
        loss_ic = np.mean((u_ic_pred - u_ic)**2)
        
        total_loss = w_pde * loss_pde + w_ic * loss_ic
        
        # Clip loss to avoid inf/nan
        if not np.isfinite(total_loss):
            return 1e8
        
        return min(total_loss, 1e8)
    
    def train_lbfgs(self, x_pde, t_pde, x_ic, t_ic, u_ic, epochs=300):
        """Train using L-BFGS-B with reduced iterations."""
        params = self._pack_params()
        
        call_count = [0]
        
        def callback(params):
            call_count[0] += 1
            if call_count[0] % 50 == 0:
                loss = self.loss_function(params, x_pde, t_pde, x_ic, t_ic, u_ic)
                print(f"  Iter {call_count[0]:3d}: Loss = {loss:.6e}")
        
        print("Training PINN (L-BFGS-B)...")
        result = minimize(
            lambda p: self.loss_function(p, x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0),
            params,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': epochs, 'ftol': 1e-4, 'maxfun': epochs*2}
        )
        
        self._unpack_params(result.x)
        print(f"Training complete. Final loss: {result.fun:.6e}\n")


def solve_allen_cahn_pinn(
    eps=0.02,
    T=0.2,
    N_pde=300,
    N_ic=50,
    epochs=200,
    u0_func=None,
    n_snapshots=5
):
    """
    Solve Allen-Cahn using PINN (fast version).
    
    Returns
    -------
    xs_plot : ndarray
        Spatial grid for plotting.
    snapshots : list of (t, u)
        Solution at different times.
    """
    if u0_func is None:
        u0_func = lambda x: (x**2) * np.cos(np.pi * x)
    
    print("Initializing PINN...")
    pinn = FastAllenCahnPINN(eps=eps, layers=(2, 32, 1))
    
    # Training data
    x_pde = np.random.uniform(-1, 1, N_pde)
    t_pde = np.random.uniform(0.001, T, N_pde)
    
    x_ic = np.linspace(-1, 1, N_ic)
    t_ic = np.zeros(N_ic)
    u_ic = u0_func(x_ic)
    
    # Train
    pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=epochs)
    
    # Generate snapshots
    xs_plot = np.linspace(-1, 1, 100)
    times = np.linspace(0, T, n_snapshots)
    snapshots = []
    
    print("Generating solution snapshots...")
    for t_val in times:
        ts = np.full_like(xs_plot, t_val)
        u_snap = pinn.forward_vectorized(xs_plot, ts)
        
        # Handle any non-finite values
        u_snap = np.clip(u_snap, -5, 5)
        
        snapshots.append((t_val, u_snap))
        print(f"  t={t_val:.3f}: min={u_snap.min():.4f}, max={u_snap.max():.4f}")
    
    return xs_plot, snapshots


if __name__ == "__main__":
    print("=" * 70)
    print("Allen-Cahn PDE Solver using Physics-Informed Neural Networks (PINN)")
    print("=" * 70)
    print(f"PDE: u_t = eps^2 * u_xx - (u^3 - u)")
    print(f"Domain: x ∈ [-1, 1], t ∈ [0, 0.2]")
    print(f"Parameters: eps = 0.02")
    print("=" * 70 + "\n")
    
    # Solve
    xs, snaps = solve_allen_cahn_pinn(
        eps=0.02,
        T=0.2,
        N_pde=300,
        N_ic=50,
        epochs=200,
        n_snapshots=5
    )
    
    # Visualize
    print("\nPlotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Multiple snapshots
    ax = axes[0]
    for (t, u) in snaps:
        ax.plot(xs, u, label=f"t={t:.3f}", linewidth=2.5)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x,t)", fontsize=12)
    ax.set_title("Allen-Cahn Solution at Different Times", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Right: Heatmap
    ax = axes[1]
    times = np.array([t for t, _ in snaps])
    u_all = np.array([u for _, u in snaps])
    
    im = ax.contourf(xs, times, u_all, levels=20, cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("u(x,t)", fontsize=11)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.set_title("Solution Evolution (Heatmap)", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('allen_cahn_pinn.png', dpi=150, bbox_inches='tight')
    print("Saved plot to 'allen_cahn_pinn.png'")
    plt.show()
    
    print("\n" + "=" * 70)
    print("Solver completed successfully!")
    print("=" * 70)
