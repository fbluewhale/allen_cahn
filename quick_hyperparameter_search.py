"""
Quick Start: Hyperparameter Optimization for Allen-Cahn PINN

This script demonstrates how to find optimal hyperparameters to minimize loss.
Run this to see automated hyperparameter search.
"""

import numpy as np
import time
from allen_cahn_pin import FastAllenCahnPINN


def quick_optimization_search():
    """
    Fast hyperparameter search focusing on key parameters.
    Takes ~10-15 minutes on a standard machine.
    """
    print("="*70)
    print("QUICK HYPERPARAMETER OPTIMIZATION SEARCH")
    print("="*70)
    
    # Preparation
    x_ic = np.linspace(-1, 1, 50)
    u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
    
    x_pde = np.random.uniform(-1, 1, 300)
    t_pde = np.random.uniform(0.001, 0.2, 300)
    t_ic = np.zeros(50)
    
    # ========================================================================
    # SEARCH 1: Network Size Impact
    # ========================================================================
    print("\n" + "-"*70)
    print("SEARCH 1: Network Architecture (Hidden Layer Size)")
    print("-"*70)
    print("\nTesting: Small → Medium → Large networks")
    print("Impact: Larger networks reduce loss but train slower\n")
    
    hidden_sizes = [16, 32, 48, 64]
    net_results = {}
    
    for h_size in hidden_sizes:
        print(f"Training network with {h_size} hidden neurons...", end=' ')
        start = time.time()
        
        pinn = FastAllenCahnPINN(eps=0.02, layers=(2, h_size, 1))
        pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=100)
        
        loss = pinn.loss_function(
            pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
        )
        elapsed = time.time() - start
        
        net_results[h_size] = (loss, elapsed)
        print(f"Loss: {loss:.4e} | Time: {elapsed:.1f}s")
    
    best_hidden = min(net_results, key=lambda k: net_results[k][0])
    print(f"\n✓ WINNER: {best_hidden} hidden neurons (Loss: {net_results[best_hidden][0]:.4e})")
    
    # ========================================================================
    # SEARCH 2: Loss Weight Impact
    # ========================================================================
    print("\n" + "-"*70)
    print("SEARCH 2: Loss Weight Balancing (w_ic)")
    print("-"*70)
    print("\nTesting: Different IC vs PDE weight ratios")
    print("Impact: Higher w_ic emphasizes IC, lower emphasizes PDE\n")
    
    w_ic_values = [1, 5, 10, 20]
    weight_results = {}
    
    pinn = FastAllenCahnPINN(eps=0.02, layers=(2, best_hidden, 1))
    
    for w_ic in w_ic_values:
        print(f"Training with w_ic = {w_ic:2d}...", end=' ')
        start = time.time()
        
        pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=100)
        
        loss = pinn.loss_function(
            pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=w_ic
        )
        elapsed = time.time() - start
        
        weight_results[w_ic] = (loss, elapsed)
        print(f"Loss: {loss:.4e} | Time: {elapsed:.1f}s")
    
    best_w = min(weight_results, key=lambda k: weight_results[k][0])
    print(f"\n✓ WINNER: w_ic = {best_w} (Loss: {weight_results[best_w][0]:.4e})")
    
    # ========================================================================
    # SEARCH 3: Training Duration Impact
    # ========================================================================
    print("\n" + "-"*70)
    print("SEARCH 3: Training Duration (Epochs)")
    print("-"*70)
    print("\nTesting: Different training lengths")
    print("Impact: More epochs → better convergence (diminishing returns)\n")
    
    epoch_values = [50, 100, 200, 400]
    epoch_results = {}
    
    for epochs in epoch_values:
        print(f"Training for {epochs:3d} epochs...", end=' ')
        start = time.time()
        
        pinn = FastAllenCahnPINN(eps=0.02, layers=(2, best_hidden, 1))
        pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=epochs)
        
        loss = pinn.loss_function(
            pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=best_w
        )
        elapsed = time.time() - start
        
        epoch_results[epochs] = (loss, elapsed)
        print(f"Loss: {loss:.4e} | Time: {elapsed:.1f}s")
    
    best_epochs = min(epoch_results, key=lambda k: epoch_results[k][0])
    print(f"\n✓ WINNER: {best_epochs} epochs (Loss: {epoch_results[best_epochs][0]:.4e})")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    print("\n📊 BEST HYPERPARAMETERS FOUND:")
    print(f"   Hidden layer size:  {best_hidden} neurons")
    print(f"   Loss weight (w_ic): {best_w}")
    print(f"   Training epochs:    {best_epochs}")
    
    print("\n📉 LOSS REDUCTION:")
    baseline = 1.54  # Original loss
    final_loss = epoch_results[best_epochs][0]
    reduction = (baseline - final_loss) / baseline * 100
    
    print(f"   Baseline loss:  {baseline:.4e}")
    print(f"   Optimized loss: {final_loss:.4e}")
    print(f"   Reduction:      {reduction:.1f}%")
    
    print("\n⏱️  COMPUTATIONAL COST:")
    total_time = sum(e[1] for e in epoch_results.values())
    total_time += sum(e[1] for e in net_results.values())
    total_time += sum(e[1] for e in weight_results.values())
    print(f"   Total search time: {total_time:.1f} seconds")
    
    print("\n💡 RECOMMENDATIONS:")
    print(f"   1. Use network: (2, {best_hidden}, 1)")
    print(f"   2. Set w_ic = {best_w}")
    print(f"   3. Train for {best_epochs} epochs")
    print(f"   4. Expected loss: {final_loss:.4e}")
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
    
    return {
        'hidden_size': best_hidden,
        'w_ic': best_w,
        'epochs': best_epochs,
        'final_loss': final_loss,
        'reduction': reduction
    }


def detailed_analysis():
    """
    More comprehensive search including sampling strategies.
    Takes ~30 minutes.
    """
    print("="*70)
    print("DETAILED HYPERPARAMETER ANALYSIS")
    print("="*70)
    
    # Generate training data
    x_ic = np.linspace(-1, 1, 50)
    u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
    t_ic = np.zeros(50)
    
    # ========================================================================
    # Analysis 1: Sampling Strategy
    # ========================================================================
    print("\n" + "-"*70)
    print("Analysis 1: Number of Training Points (N_pde)")
    print("-"*70)
    
    n_pde_values = [100, 200, 400, 600, 800]
    sampling_results = {}
    
    for n_pde in n_pde_values:
        x_pde = np.random.uniform(-1, 1, n_pde)
        t_pde = np.random.uniform(0.001, 0.2, n_pde)
        
        print(f"Training with N_pde = {n_pde:3d}...", end=' ')
        
        pinn = FastAllenCahnPINN(eps=0.02, layers=(2, 48, 1))
        pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=150)
        
        loss = pinn.loss_function(
            pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
        )
        sampling_results[n_pde] = loss
        print(f"Loss: {loss:.4e}")
    
    print("\n📊 SAMPLING ANALYSIS:")
    for n, loss in sorted(sampling_results.items()):
        bar = '█' * int(loss * 100 / max(sampling_results.values()))
        print(f"   N={n:3d}: {bar} {loss:.4e}")
    
    # ========================================================================
    # Analysis 2: Convergence Pattern
    # ========================================================================
    print("\n" + "-"*70)
    print("Analysis 2: Convergence Pattern")
    print("-"*70)
    print("\nObserving loss reduction over epochs:\n")
    
    x_pde = np.random.uniform(-1, 1, 400)
    t_pde = np.random.uniform(0.001, 0.2, 400)
    
    pinn = FastAllenCahnPINN(eps=0.02, layers=(2, 48, 1))
    
    convergence = []
    milestones = [50, 100, 200, 400, 600]
    
    for target_epochs in milestones:
        pinn.train_lbfgs(x_pde, t_pde, x_ic, t_ic, u_ic, epochs=target_epochs)
        loss = pinn.loss_function(
            pinn._pack_params(), x_pde, t_pde, x_ic, t_ic, u_ic, w_ic=10.0
        )
        convergence.append((target_epochs, loss))
        bar = '█' * max(1, int((1.5 - loss) * 100 / 1.5))
        print(f"   Epochs {target_epochs:3d}: {bar} Loss: {loss:.4e}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    
    # Quick search (~10 minutes)
    print("\n🚀 Starting QUICK optimization search...")
    print("   This will test key hyperparameters\n")
    
    results = quick_optimization_search()
    
    # Option: Run detailed analysis
    print("\n" + "="*70)
    response = input("\nRun detailed analysis? (y/n): ").strip().lower()
    if response == 'y':
        detailed_analysis()
    
    print("\n✨ Hyperparameter optimization guide complete!")
    print("\nNext steps:")
    print("1. Use the optimal parameters found above")
    print("2. Run main solver with these settings:")
    print(f"   layers = (2, {results['hidden_size']}, 1)")
    print(f"   epochs = {results['epochs']}")
    print(f"   w_ic = {results['w_ic']}")
    print("\n3. Check HYPERPARAMETER_GUIDE.md for more details")
