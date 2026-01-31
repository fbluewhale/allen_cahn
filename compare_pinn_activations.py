"""
Comparison of Allen-Cahn PINN with Jacobi activation vs normal PINN (ReLU/Tanh).

This script solves the Allen-Cahn equation using two different activation functions
and compares their performance in terms of convergence, accuracy, and solution quality.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_jacobi
import matplotlib.pyplot as plt
import time


class AllenCahnPINN:
    """Generic PINN for Allen-Cahn with configurable activation function."""
    
    def __init__(self, eps=0.01, layers=(2, 96, 1), activation='relu'):
        """
        Initialize the PINN.
        
        Parameters
        ----------
        eps : float
            Interface parameter.
        layers : tuple
            Layer sizes (input, hidden..., output).
        activation : str
            'relu', 'tanh', or 'jacobi'
        """
        self.eps = eps
        self.layers = layers
        self.activation = activation
        self.training_losses = []
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
    
    def jacobi_activation(self, x):
        """Jacobi polynomial activation: P_3^(0,0)(x)"""
        x_clipped = np.clip(x, -1.0, 1.0)
        return eval_jacobi(3, 0, 0, x_clipped)
    
    def tanh_activation(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def relu_activation(self, x):
        """ReLU activation function."""
        return np.maximum(x, 0.0)
    
    def apply_activation(self, z):
        """Apply the selected activation function."""
        if self.activation == 'jacobi':
            return np.vectorize(self.jacobi_activation)(z)
        elif self.activation == 'tanh':
            return self.tanh_activation(z)
        else:  # relu
            return self.relu_activation(z)
    
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
        x_n = np.clip(x, -1.0, 1.0)
        t_n = np.clip(t, 0.0, 1.0)
        z = np.array([[x_n, t_n]])
        
        # Hidden layers with selected activation
        for i in range(len(self.W) - 1):
            z = np.dot(z, self.W[i]) + self.b[i]
            z = self.apply_activation(z)
        
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
        
        # PDE loss
        residuals = np.array([self.pde_residual(x_pde[i], t_pde[i]) for i in range(len(x_pde))])
        loss_pde = np.mean(residuals**2)
        
        # IC loss
        u_ic_pred = self.forward_vectorized(x_ic, t_ic)
        loss_ic = np.mean((u_ic_pred - u_ic)**2)
        
        total_loss = w_pde * loss_pde + w_ic * loss_ic
        
        # Track loss
        self.training_losses.append(total_loss)
        
        if not np.isfinite(total_loss):
            return 1e8
        
        return min(total_loss, 1e8)
    
    def train_lbfgs(self, x_pde, t_pde, x_ic, t_ic, u_ic, epochs=300):
        """Train using L-BFGS-B."""
        params = self._pack_params()
        
        call_count = [0]
        
        def callback(params):
            call_count[0] += 1
            if call_count[0] % 50 == 0:
                loss = self.loss_function(params, x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=20.0)
                print(f"    Iter {call_count[0]:3d}: Loss = {loss:.6e}")
        
        result = minimize(
            lambda p: self.loss_function(p, x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=20.0),
            params,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': epochs, 'ftol': 1e-4, 'maxfun': epochs*2}
        )
        
        self._unpack_params(result.x)
        return result.fun


def solve_allen_cahn_comparison(
    eps=0.02,
    T=0.2,
    N_pde=800,
    N_ic=150,
    epochs=600,
    activations=['relu', 'tanh', 'jacobi']
):
    """
    Solve Allen-Cahn using multiple activation functions and compare results.
    
    Returns
    -------
    results : dict
        Dictionary with results for each activation function.
    """
    
    u0_func = lambda x: (x**2) * np.cos(np.pi * x)
    
    # Training data
    x_pde = np.random.uniform(-1, 1, N_pde)
    t_pde = np.random.uniform(0.001, T, N_pde)
    
    x_ic = np.linspace(-1, 1, N_ic)
    t_ic = np.zeros(N_ic)
    u_ic = u0_func(x_ic)
    
    results = {}
    
    for activation in activations:
        print(f"\n{'='*70}")
        print(f"Training PINN with {activation.upper()} activation")
        print(f"{'='*70}")
        
        pinn = AllenCahnPINN(eps=eps, layers=(2, 48, 1), activation=activation)
        
        start_time = time.time()
        final_loss = pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=epochs)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Final loss: {final_loss:.6e}")
        
        # Generate snapshots
        xs_plot = np.linspace(-1, 1, 100)
        times = np.linspace(0, T, 10)
        snapshots = []
        
        for t_val in times:
            ts = np.full_like(xs_plot, t_val)
            u_snap = pinn.forward_vectorized(xs_plot, ts)
            u_snap = np.clip(u_snap, -5, 5)
            snapshots.append((t_val, u_snap))
        
        results[activation] = {
            'pinn': pinn,
            'xs': xs_plot,
            'snapshots': snapshots,
            'final_loss': final_loss,
            'training_time': training_time,
            'training_losses': pinn.training_losses
        }
    
    return results, x_ic, u_ic


def compute_metrics(results):
    """Compute comparison metrics."""
    metrics = {}
    
    for activation, data in results.items():
        pinn = data['pinn']
        
        # IC accuracy
        u_ic_pred = pinn.forward_vectorized(np.linspace(-1, 1, 150), np.zeros(150))
        u_ic_true = (np.linspace(-1, 1, 150)**2) * np.cos(np.pi * np.linspace(-1, 1, 150))
        ic_error = np.mean((u_ic_pred - u_ic_true)**2)
        
        metrics[activation] = {
            'final_loss': data['final_loss'],
            'training_time': data['training_time'],
            'ic_error': ic_error,
            'min_loss': min(data['training_losses']),
            'mean_loss': np.mean(data['training_losses'][-50:])
        }
    
    return metrics


def plot_comparison(results, x_ic, u_ic):
    """Create comprehensive comparison plots."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    activations = list(results.keys())
    colors = {'relu': 'blue', 'tanh': 'green', 'jacobi': 'red'}
    
    # Row 1: Solutions at different times
    for col, activation in enumerate(activations):
        ax = fig.add_subplot(gs[0, col])
        xs = results[activation]['xs']
        for t, u in results[activation]['snapshots'][::3]:
            ax.plot(xs, u, linewidth=2)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x,t)', fontsize=10)
        ax.set_title(f'{activation.upper()} - Solutions', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Row 2: Training loss convergence
    ax = fig.add_subplot(gs[1, :])
    for activation in activations:
        losses = results[activation]['training_losses']
        ax.semilogy(losses, label=activation.upper(), linewidth=2.5, 
                   color=colors.get(activation, 'black'))
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Heatmaps
    for col, activation in enumerate(activations):
        ax = fig.add_subplot(gs[2, col])
        xs = results[activation]['xs']
        times = np.array([t for t, _ in results[activation]['snapshots']])
        u_all = np.array([u for _, u in results[activation]['snapshots']])
        
        im = ax.contourf(xs, times, u_all, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('t', fontsize=10)
        ax.set_title(f'{activation.upper()} - Evolution', fontsize=11, fontweight='bold')
    
    plt.suptitle('Allen-Cahn PINN Comparison: Jacobi vs Standard Activations', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('pinn_activation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to 'pinn_activation_comparison.png'")
    plt.show()


def plot_metrics_comparison(metrics):
    """Plot performance metrics comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    activations = list(metrics.keys())
    x_pos = np.arange(len(activations))
    
    # Final loss
    ax = axes[0, 0]
    final_losses = [metrics[a]['final_loss'] for a in activations]
    bars = ax.bar(x_pos, final_losses, color=['blue', 'green', 'red'][:len(activations)])
    ax.set_ylabel('Final Loss', fontsize=11)
    ax.set_title('Final Training Loss', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training time
    ax = axes[0, 1]
    times = [metrics[a]['training_time'] for a in activations]
    bars = ax.bar(x_pos, times, color=['blue', 'green', 'red'][:len(activations)])
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.grid(True, alpha=0.3, axis='y')
    
    # IC Error
    ax = axes[1, 0]
    ic_errors = [metrics[a]['ic_error'] for a in activations]
    bars = ax.bar(x_pos, ic_errors, color=['blue', 'green', 'red'][:len(activations)])
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Initial Condition Error', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mean loss (last 50 iterations)
    ax = axes[1, 1]
    mean_losses = [metrics[a]['mean_loss'] for a in activations]
    bars = ax.bar(x_pos, mean_losses, color=['blue', 'green', 'red'][:len(activations)])
    ax.set_ylabel('Mean Loss', fontsize=11)
    ax.set_title('Mean Loss (Last 50 Iterations)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pinn_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved metrics plot to 'pinn_metrics_comparison.png'")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Allen-Cahn PINN Activation Function Comparison")
    print("=" * 70)
    print("Comparing: ReLU, Tanh, and Jacobi Polynomial Activations")
    print("=" * 70)
    
    # Solve with all three activations
    results, x_ic, u_ic = solve_allen_cahn_comparison(
        eps=0.01,
        T=0.2,
        N_pde=800,
        N_ic=100,
        epochs=500,
        activations=['relu', 'tanh', 'jacobi']
    )
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 70)
    metrics = compute_metrics(results)
    
    for activation, metric_dict in metrics.items():
        print(f"\n{activation.upper()}:")
        print(f"  Final Loss:           {metric_dict['final_loss']:.6e}")
        print(f"  Training Time:        {metric_dict['training_time']:.2f}s")
        print(f"  IC Error (MSE):       {metric_dict['ic_error']:.6e}")
        print(f"  Min Loss:             {metric_dict['min_loss']:.6e}")
        print(f"  Mean Loss (last 50):  {metric_dict['mean_loss']:.6e}")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating comparison visualizations...")
    print("=" * 70)
    plot_comparison(results, x_ic, u_ic)
    plot_metrics_comparison(metrics)
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
