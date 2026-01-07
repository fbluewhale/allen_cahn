# Hyperparameter Optimization Guide for Allen-Cahn PINN Solver

## Overview

Reducing the **loss function magnitude** requires careful tuning of hyperparameters. This guide explains:
1. **What hyperparameters matter**
2. **How to find optimal values**
3. **Practical optimization strategies**
4. **Tools for systematic search**

---

## Key Hyperparameters

### 1. **Network Architecture**

#### Current Setting
```python
layers = (2, 32, 1)  # Input → 32 hidden neurons → Output
```

#### How it affects loss
- **Larger networks** (e.g., 64, 128 neurons):
  - ✅ Better function approximation capability
  - ✅ Can reduce loss more (lower final loss)
  - ❌ Slower training
  - ❌ Risk of overfitting

- **Smaller networks** (e.g., 16 neurons):
  - ✅ Faster training
  - ❌ Limited capacity (underfitting)
  - ❌ Higher minimum achievable loss

#### Recommendation
```python
# For better loss reduction, try:
layers = (2, 64, 1)    # Moderate increase
# or
layers = (2, 128, 1)   # Larger (if time permits)
```

**Optimal range:** 32-128 hidden neurons for this problem

---

### 2. **Loss Weights** (w_pde, w_ic)

#### Current Setting
```python
loss = w_pde * loss_pde + w_ic * loss_ic
# In training: w_ic = 10.0 (IC strongly weighted)
```

#### How it affects loss
- **High w_ic** (e.g., 20, 50):
  - ✅ Initial condition satisfied well
  - ❌ PDE residual may grow
  - Result: Total loss appears lower but solution less physical

- **Low w_ic** (e.g., 1, 5):
  - ✅ Better PDE satisfaction
  - ❌ Initial condition may be violated
  - Result: Loss higher, but better physics

#### Recommendation
```python
# Balance approach:
w_ic = 10.0      # Current (good balance)

# If IC fitting is priority:
w_ic = 20.0      # More weight to IC

# If PDE accuracy is priority:
w_ic = 5.0       # Equal weight to both
```

**Optimal range:** 5-20 (depends on problem importance)

---

### 3. **Number of Training Points**

#### Current Setting
```python
N_pde = 300      # Interior PDE residual points
N_ic = 50        # Initial condition points
```

#### How it affects loss
- **More points** (N_pde = 500-1000):
  - ✅ Better coverage of domain
  - ✅ Typically lower final loss
  - ❌ Slower training
  - ❌ More evaluations needed

- **Fewer points** (N_pde = 100):
  - ✅ Fast training
  - ❌ Poor coverage
  - ❌ Higher loss

#### Recommendation
```python
# For lower loss:
N_pde = 600      # Double the coverage
N_ic = 100       # More IC points

# For fast experiments:
N_pde = 200
N_ic = 30
```

**Optimal range:** N_pde ∈ [300, 800], N_ic ∈ [50, 150]

---

### 4. **Finite Difference Step Size** (h)

#### Current Setting
```python
h = 0.005  # For computing u_t, u_xx numerically
```

#### How it affects loss
- **Too small h** (e.g., 0.001):
  - ✅ Theoretically more accurate
  - ❌ Numerical round-off errors dominate
  - Result: Noisy derivatives

- **Too large h** (e.g., 0.05):
  - ✅ Averages out noise
  - ❌ Truncation errors dominate
  - Result: Inaccurate derivatives

#### Theory (Richardson extrapolation)
$$\text{Optimal } h \approx \sqrt[3]{\varepsilon_m} \times \text{scale}$$

where $\varepsilon_m$ ≈ 10^-16 (machine epsilon)

#### Recommendation
```python
# Current is good (h = 0.005)
# Try these if loss is high:
h = 0.01    # If noisy derivatives
h = 0.002   # If want more accuracy
```

**Optimal range:** h ∈ [0.002, 0.02]

---

### 5. **Training Epochs**

#### Current Setting
```python
epochs = 200  # L-BFGS-B iterations
```

#### How it affects loss
- **More epochs** (500, 1000):
  - ✅ Better convergence
  - ✅ Lower final loss (diminishing returns)
  - ❌ Slower training

- **Too few epochs** (50):
  - ✅ Fast
  - ❌ Premature stopping (high loss)

#### Loss convergence pattern
```
Loss vs Epochs (typical):
┌─
│  \
│   \___
│       \____
│           \___  ← Diminishing returns
└─────────────────
0      100    200    400
```

#### Recommendation
```python
# Monitor and decide:
epochs = 200   # Current (good sweet spot)
epochs = 400   # If loss plateau not reached
```

**Optimal range:** 200-500 epochs

---

## Optimization Strategies

### Strategy 1: Grid Search (Systematic)

Test combinations of key hyperparameters:

```python
from hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(eps=0.02, T=0.2)

# Test different network sizes
net_results = optimizer.optimize_network_size([16, 32, 48, 64])

# Test different loss weights
weight_results = optimizer.optimize_loss_weights([1, 5, 10, 20])

# Test different training durations
epoch_results = optimizer.optimize_epochs([100, 200, 400, 800])

# View results
optimizer.report()
optimizer.plot_results()
```

**Time complexity:** Medium (minutes to hours)
**Accuracy:** Good
**Recommendation:** ✅ **Best for systematic search**

---

### Strategy 2: Random Search

Sample hyperparameters randomly (fast exploration):

```python
import numpy as np

hyperparams = []
for _ in range(20):
    hidden_size = np.random.choice([16, 32, 64, 128])
    w_ic = np.random.uniform(1, 50)
    n_pde = np.random.choice([200, 400, 600, 800])
    
    hyperparams.append({
        'layers': (2, hidden_size, 1),
        'w_ic': w_ic,
        'n_pde': n_pde
    })
    # Train and record loss for each
```

**Time complexity:** Fast
**Accuracy:** Fair
**Use when:** Quick exploration needed

---

### Strategy 3: Bayesian Optimization

Use probabilistic model to guide search (recommended for expensive functions):

```python
from skopt import gp_minimize

def objective(params):
    hidden_size, w_ic, n_pde = params
    
    # Train PINN with these hyperparameters
    pinn = FastAllenCahnPINN(eps=0.02, layers=(2, int(hidden_size), 1))
    # ... train and return loss
    
    return loss

# Run Bayesian optimization
result = gp_minimize(
    objective,
    dimensions=[(16, 128), (1, 50), (200, 800)],  # Parameter ranges
    n_calls=30,
    n_initial_points=5
)
```

**Time complexity:** Fast + Medium
**Accuracy:** Excellent
**Recommendation:** ✅ **Best for expensive optimization**

---

## Quick Optimization Steps

### Step 1: Get Baseline Loss
```python
xs, snaps = solve_allen_cahn_pinn(
    eps=0.02, T=0.2, N_pde=300, N_ic=50, epochs=200
)
# Current loss: ~1.54
```

### Step 2: Increase Network Size
```python
xs, snaps = solve_allen_cahn_pinn(
    eps=0.02, T=0.2, N_pde=300, N_ic=50, epochs=200,
    network_params={'layers': (2, 64, 1)}  # Increase hidden neurons
)
# Expected: Loss → ~1.2-1.3
```

### Step 3: Increase Training
```python
xs, snaps = solve_allen_cahn_pinn(
    eps=0.02, T=0.2, N_pde=300, N_ic=50, epochs=500
)
# Expected: Loss → ~1.0-1.2
```

### Step 4: Add More Training Points
```python
xs, snaps = solve_allen_cahn_pinn(
    eps=0.02, T=0.2, N_pde=600, N_ic=100, epochs=500
)
# Expected: Loss → ~0.8-1.0
```

### Step 5: Fine-tune Weights
```python
# Adjust w_ic based on need
xs, snaps = solve_allen_cahn_pinn(
    eps=0.02, T=0.2, N_pde=600, N_ic=100, epochs=500,
    w_ic=15  # Increase IC weighting
)
```

---

## Loss Reduction Targets

Starting from baseline loss ≈ **1.54**:

| Action | Expected Loss | Comments |
|--------|---------------|----------|
| Increase hidden: 32→64 | 1.2-1.3 | +10-20% improvement |
| Increase epochs: 200→500 | 1.0-1.2 | +20-30% improvement |
| More points: 300→600 | 0.8-1.0 | +30-40% improvement |
| Optimize w_ic: 10→15 | 0.7-0.9 | +30-50% improvement |
| **All combined** | **0.5-0.7** | ✅ **50-60% reduction** |

---

## Advanced Tips

### 1. **Learning Rate Scheduling**
L-BFGS-B doesn't have learning rate, but you can use other optimizers:

```python
from scipy.optimize import minimize

# Adam-like approach with learning rate decay
for epoch in range(epochs):
    lr = initial_lr * (1 - epoch/epochs)  # Decay
    # Update with lr
```

### 2. **Batch Normalization**
Add layer normalization:

```python
class NormalizedPINN(FastAllenCahnPINN):
    def forward_single(self, x, t):
        # ... existing forward pass ...
        # Add normalization after each layer
        z = (z - z.mean()) / (z.std() + 1e-6)
```

### 3. **Early Stopping**
Stop when loss plateaus:

```python
def train_with_early_stopping(self, ..., patience=50):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        loss = self.loss_function(...)
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
```

### 4. **Adaptive Sampling**
Focus on regions with high error:

```python
def adaptive_sampling(self, residuals, n_samples):
    # Sample more from high-residual regions
    weights = np.abs(residuals) / np.abs(residuals).sum()
    indices = np.random.choice(
        len(residuals), n_samples, p=weights
    )
    return indices
```

---

## Practical Workflow

```python
from hyperparameter_optimizer import HyperparameterOptimizer
import matplotlib.pyplot as plt

# 1. Create optimizer
opt = HyperparameterOptimizer(eps=0.02, T=0.2)

# 2. Run grid search (picks best parameters automatically)
print("Phase 1: Network size search...")
net_results, times = opt.optimize_network_size([32, 48, 64, 96])

print("Phase 2: Loss weight search...")
weight_results = opt.optimize_loss_weights([5, 10, 15, 20])

print("Phase 3: Training duration search...")
epoch_results, times = opt.optimize_epochs([200, 300, 400, 500])

# 3. Generate comprehensive report
opt.report()
opt.plot_results('optimization_results.png')

# 4. Print recommendations
print("\n✅ OPTIMIZATION COMPLETE")
print("See 'optimization_results.png' for detailed visualization")
```

---

## Summary Table

| Parameter | Current | Good Range | Effect on Loss |
|-----------|---------|-----------|-----------------|
| Hidden neurons | 32 | 32-128 | ↓ Higher = lower loss |
| w_ic | 10 | 5-20 | Depends on balance |
| N_pde | 300 | 300-800 | ↓ Higher = lower loss |
| N_ic | 50 | 50-150 | ↓ Higher = lower loss |
| Epochs | 200 | 200-500 | ↓ Higher = lower loss |
| h (FD step) | 0.005 | 0.002-0.02 | ~ Optimal ≈ 0.005-0.01 |

---

## Files Provided

- **`hyperparameter_optimizer.py`** - Full optimization toolkit
- **`allen_cahn_pin.py`** - Main PINN solver
- **This guide** - Comprehensive documentation

Use `hyperparameter_optimizer.py` to systematically find best parameters!
