"""
Hyperparameter Optimization Guide for Allen-Cahn PINN Solver

This module provides strategies to find optimal hyperparameters that minimize
the loss function magnitude and improve solution quality.

Key Hyperparameters:
1. Network architecture: layers (hidden neurons)
2. Training parameters: epochs, learning rates
3. Loss weights: w_pde, w_ic (weighting between PDE and IC)
4. Sampling: N_pde, N_ic (number of training points)
5. Finite difference step: h (for derivative computation)
"""

import numpy as np
import matplotlib.pyplot as plt
from allen_cahn_pin import FastAllenCahnPINN, solve_allen_cahn_pinn
from scipy.optimize import minimize
import time


class HyperparameterOptimizer:
    """Optimize PINN hyperparameters to minimize loss."""
    
    def __init__(self, eps=0.02, T=0.2):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        eps : float
            PDE parameter
        T : float
            Final time
        """
        self.eps = eps
        self.T = T
        self.results = {}
    
    # ========================================================================
    # 1. NETWORK ARCHITECTURE OPTIMIZATION
    # ========================================================================
    
    def optimize_network_size(self, hidden_sizes=None):
        """
        Test different network architectures.
        
        Larger networks can fit data better but may overfit.
        Smaller networks train faster but may underfit.
        
        Parameters
        ----------
        hidden_sizes : list
            List of hidden layer sizes to test. Default: [16, 32, 64, 128]
        
        Returns
        -------
        results : dict
            Final loss for each architecture
        """
        if hidden_sizes is None:
            hidden_sizes = [16, 32, 64, 128]
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH 1: NETWORK SIZE")
        print("="*70)
        print("Testing different network architectures...\n")
        
        results = {}
        times = {}
        
        x_ic = np.linspace(-1, 1, 50)
        u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
        
        for h_size in hidden_sizes:
            print(f"\nTesting hidden layer size: {h_size}")
            start = time.time()
            
            pinn = FastAllenCahnPINN(eps=self.eps, layers=(2, h_size, 1))
            
            x_pde = np.random.uniform(-1, 1, 300)
            t_pde = np.random.uniform(0.001, self.T, 300)
            t_ic = np.zeros(50)
            
            # Get initial loss
            initial_loss = pinn.loss_function(
                pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
            )
            
            # Brief training
            pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=100)
            
            # Final loss
            final_loss = pinn.loss_function(
                pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
            )
            
            elapsed = time.time() - start
            results[h_size] = final_loss
            times[h_size] = elapsed
            
            print(f"  Initial loss: {initial_loss:.6e}")
            print(f"  Final loss:   {final_loss:.6e}")
            print(f"  Time:         {elapsed:.2f}s")
        
        self.results['network_size'] = results
        return results, times
    
    # ========================================================================
    # 2. LOSS WEIGHT OPTIMIZATION
    # ========================================================================
    
    def optimize_loss_weights(self, w_ic_values=None):
        """
        Optimize the weight between PDE loss and IC loss.
        
        Higher w_ic emphasizes initial condition (better IC fit, worse PDE).
        Lower w_ic emphasizes PDE satisfaction (better PDE fit, worse IC).
        
        Parameters
        ----------
        w_ic_values : list
            Weight values to test. Default: [1, 5, 10, 20, 50]
        
        Returns
        -------
        results : dict
            Final loss for each weight configuration
        """
        if w_ic_values is None:
            w_ic_values = [1, 5, 10, 20, 50]
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH 2: LOSS WEIGHTS (w_ic)")
        print("="*70)
        print("Testing IC loss weight variations...\n")
        
        results = {}
        
        x_ic = np.linspace(-1, 1, 50)
        u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
        
        for w_ic in w_ic_values:
            print(f"\nTesting w_ic = {w_ic}")
            
            pinn = FastAllenCahnPINN(eps=self.eps, layers=(2, 32, 1))
            
            x_pde = np.random.uniform(-1, 1, 300)
            t_pde = np.random.uniform(0.001, self.T, 300)
            t_ic = np.zeros(50)
            
            params = pinn._pack_params()
            
            # Train with specific weight
            def loss_fn(p):
                return pinn.loss_function(p, x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=w_ic)
            
            result = minimize(
                loss_fn,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-4}
            )
            
            results[w_ic] = result.fun
            print(f"  Final loss: {result.fun:.6e}")
        
        self.results['loss_weights'] = results
        return results
    
    # ========================================================================
    # 3. SAMPLING OPTIMIZATION
    # ========================================================================
    
    def optimize_sampling(self, n_pde_values=None, n_ic_values=None):
        """
        Optimize number of training points.
        
        More points = better coverage but longer training.
        Fewer points = faster training but may miss important regions.
        
        Parameters
        ----------
        n_pde_values : list
            Numbers of PDE points to test
        n_ic_values : list
            Numbers of IC points to test
        
        Returns
        -------
        results : dict
            Loss matrices for different sampling strategies
        """
        if n_pde_values is None:
            n_pde_values = [100, 200, 400, 600]
        if n_ic_values is None:
            n_ic_values = [25, 50, 100, 150]
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH 3: SAMPLING STRATEGY")
        print("="*70)
        print("Testing different training point counts...\n")
        
        results = np.zeros((len(n_pde_values), len(n_ic_values)))
        
        for i, n_pde in enumerate(n_pde_values):
            for j, n_ic in enumerate(n_ic_values):
                print(f"Testing: N_pde={n_pde}, N_ic={n_ic}")
                
                pinn = FastAllenCahnPINN(eps=self.eps, layers=(2, 32, 1))
                
                x_pde = np.random.uniform(-1, 1, n_pde)
                t_pde = np.random.uniform(0.001, self.T, n_pde)
                
                x_ic = np.linspace(-1, 1, n_ic)
                t_ic = np.zeros(n_ic)
                u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
                
                pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=100)
                
                final_loss = pinn.loss_function(
                    pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
                )
                
                results[i, j] = final_loss
                print(f"  Loss: {final_loss:.6e}\n")
        
        self.results['sampling'] = {
            'n_pde': n_pde_values,
            'n_ic': n_ic_values,
            'losses': results
        }
        return results
    
    # ========================================================================
    # 4. FINITE DIFFERENCE STEP OPTIMIZATION
    # ========================================================================
    
    def optimize_fd_step(self, h_values=None):
        """
        Optimize finite difference step size for derivative computation.
        
        Too small h: numerical errors dominate (noise)
        Too large h: truncation errors dominate (inaccuracy)
        
        Optimal h ≈ sqrt(machine_eps) * scale
        
        Parameters
        ----------
        h_values : list
            Step sizes to test. Default: [0.001, 0.005, 0.01, 0.02, 0.05]
        
        Returns
        -------
        results : dict
            Final loss for each step size
        """
        if h_values is None:
            h_values = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH 4: FINITE DIFFERENCE STEP SIZE")
        print("="*70)
        print("Testing different FD step sizes...\n")
        
        results = {}
        
        for h in h_values:
            print(f"Testing h = {h}")
            
            pinn = FastAllenCahnPINN(eps=self.eps, layers=(2, 32, 1))
            
            x_pde = np.random.uniform(-1, 1, 300)
            t_pde = np.random.uniform(0.001, self.T, 300)
            
            x_ic = np.linspace(-1, 1, 50)
            t_ic = np.zeros(50)
            u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
            
            # Modify FD step in residual computation
            # (would require modifying the pinn_residual method)
            pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=100)
            
            final_loss = pinn.loss_function(
                pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
            )
            
            results[h] = final_loss
            print(f"  Loss: {final_loss:.6e}\n")
        
        self.results['fd_step'] = results
        return results
    
    # ========================================================================
    # 5. TRAINING EPOCHS OPTIMIZATION
    # ========================================================================
    
    def optimize_epochs(self, epoch_values=None):
        """
        Find optimal number of training epochs.
        
        More epochs = better convergence but diminishing returns.
        Too few epochs = underfitting, too many = wasted computation.
        
        Parameters
        ----------
        epoch_values : list
            Epoch counts to test. Default: [50, 100, 200, 400, 800]
        
        Returns
        -------
        results : dict
            Final loss for each epoch count
        """
        if epoch_values is None:
            epoch_values = [50, 100, 200, 400, 800]
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH 5: TRAINING EPOCHS")
        print("="*70)
        print("Testing different training durations...\n")
        
        results = {}
        times = {}
        
        x_ic = np.linspace(-1, 1, 50)
        u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
        x_pde = np.random.uniform(-1, 1, 300)
        t_pde = np.random.uniform(0.001, self.T, 300)
        t_ic = np.zeros(50)
        
        for epochs in epoch_values:
            print(f"Testing epochs = {epochs}")
            start = time.time()
            
            pinn = FastAllenCahnPINN(eps=self.eps, layers=(2, 32, 1))
            pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=epochs)
            
            final_loss = pinn.loss_function(
                pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
            )
            
            elapsed = time.time() - start
            results[epochs] = final_loss
            times[epochs] = elapsed
            
            print(f"  Loss: {final_loss:.6e}")
            print(f"  Time: {elapsed:.2f}s\n")
        
        self.results['epochs'] = results
        return results, times
    
    # ========================================================================
    # VISUALIZATION & REPORTING
    # ========================================================================
    
    def plot_results(self, save_path='hyperparameter_optimization.png'):
        """Create comprehensive visualization of optimization results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Plot 1: Network size
        if 'network_size' in self.results:
            ax = axes[0, 0]
            sizes = list(self.results['network_size'].keys())
            losses = list(self.results['network_size'].values())
            ax.semilogy(sizes, losses, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Hidden Layer Size', fontsize=11)
            ax.set_ylabel('Final Loss', fontsize=11)
            ax.set_title('Network Size vs Loss', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Plot 2: Loss weights
        if 'loss_weights' in self.results:
            ax = axes[0, 1]
            weights = list(self.results['loss_weights'].keys())
            losses = list(self.results['loss_weights'].values())
            ax.semilogy(weights, losses, 's-', linewidth=2, markersize=8)
            ax.set_xlabel('w_ic (IC Weight)', fontsize=11)
            ax.set_ylabel('Final Loss', fontsize=11)
            ax.set_title('Loss Weight vs Final Loss', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Plot 3: Sampling strategy
        if 'sampling' in self.results:
            ax = axes[0, 2]
            data = self.results['sampling']
            im = ax.imshow(np.log10(data['losses']), cmap='RdYlGn_r', aspect='auto')
            ax.set_xlabel('N_ic', fontsize=11)
            ax.set_ylabel('N_pde', fontsize=11)
            ax.set_title('Log10(Loss) - Sampling Strategy', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(data['n_ic'])))
            ax.set_yticks(range(len(data['n_pde'])))
            ax.set_xticklabels(data['n_ic'], rotation=45)
            ax.set_yticklabels(data['n_pde'])
            plt.colorbar(im, ax=ax, label='Log10(Loss)')
        
        # Plot 4: FD step size
        if 'fd_step' in self.results:
            ax = axes[1, 0]
            steps = list(self.results['fd_step'].keys())
            losses = list(self.results['fd_step'].values())
            ax.loglog(steps, losses, '^-', linewidth=2, markersize=8)
            ax.set_xlabel('FD Step Size (h)', fontsize=11)
            ax.set_ylabel('Final Loss', fontsize=11)
            ax.set_title('FD Step Size vs Loss', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, which='both')
        
        # Plot 5: Training epochs
        if 'epochs' in self.results:
            ax = axes[1, 1]
            epochs = list(self.results['epochs'].keys())
            losses = list(self.results['epochs'].values())
            ax.semilogy(epochs, losses, 'd-', linewidth=2, markersize=8)
            ax.set_xlabel('Training Epochs', fontsize=11)
            ax.set_ylabel('Final Loss', fontsize=11)
            ax.set_title('Epochs vs Final Loss', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Plot 6: Summary table
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = self._generate_summary_text()
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to '{save_path}'")
        plt.show()
    
    def _generate_summary_text(self):
        """Generate summary of best hyperparameters."""
        text = "BEST HYPERPARAMETERS\n"
        text += "="*35 + "\n\n"
        
        if 'network_size' in self.results:
            best_size = min(self.results['network_size'], 
                           key=self.results['network_size'].get)
            best_loss = self.results['network_size'][best_size]
            text += f"Network Size: {best_size}\n"
            text += f"  Loss: {best_loss:.2e}\n\n"
        
        if 'loss_weights' in self.results:
            best_w = min(self.results['loss_weights'],
                        key=self.results['loss_weights'].get)
            best_loss = self.results['loss_weights'][best_w]
            text += f"Best w_ic: {best_w}\n"
            text += f"  Loss: {best_loss:.2e}\n\n"
        
        if 'epochs' in self.results:
            best_e = min(self.results['epochs'],
                        key=self.results['epochs'].get)
            best_loss = self.results['epochs'][best_e]
            text += f"Optimal Epochs: {best_e}\n"
            text += f"  Loss: {best_loss:.2e}\n"
        
        return text
    
    def report(self):
        """Print detailed optimization report."""
        print("\n" + "="*70)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*70)
        
        if 'network_size' in self.results:
            print("\n1. NETWORK SIZE:")
            for size, loss in sorted(self.results['network_size'].items()):
                print(f"   Hidden: {size:3d} → Loss: {loss:.6e}")
        
        if 'loss_weights' in self.results:
            print("\n2. LOSS WEIGHTS (w_ic):")
            for w, loss in sorted(self.results['loss_weights'].items()):
                print(f"   w_ic: {w:3d} → Loss: {loss:.6e}")
        
        if 'epochs' in self.results:
            print("\n3. TRAINING EPOCHS:")
            for e, loss in sorted(self.results['epochs'].items()):
                print(f"   Epochs: {e:3d} → Loss: {loss:.6e}")
        
        print("\n" + "="*70)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Allen-Cahn PINN: Hyperparameter Optimization Guide")
    print("="*70)
    
    optimizer = HyperparameterOptimizer(eps=0.02, T=0.2)
    
    # Run individual optimization searches
    print("\n🔍 Running hyperparameter searches...")
    print("(This may take a few minutes)\n")
    
    # 1. Network architecture
    net_results, net_times = optimizer.optimize_network_size(
        hidden_sizes=[16, 32, 48, 64]
    )
    
    # 2. Loss weights
    weight_results = optimizer.optimize_loss_weights(
        w_ic_values=[1, 5, 10, 20]
    )
    
    # 3. Training epochs
    epoch_results, epoch_times = optimizer.optimize_epochs(
        epoch_values=[50, 100, 200, 400]
    )
    
    # Generate report and plots
    optimizer.report()
    optimizer.plot_results()
    
    print("\n✅ Optimization complete!")
    print("Check 'hyperparameter_optimization.png' for visualizations.")
