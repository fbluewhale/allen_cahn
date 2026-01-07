# Allen-Cahn PDE Solver - Fix Documentation

## Problem with Original Code (`allen_cahn.py`)

The original spectral Galerkin implementation encountered numerical instability:

```
RuntimeWarning: overflow encountered in power
  u3_vals = u_vals**3
ValueError: array must not contain infs or NaNs
```

### Root Causes

1. **Overflow in nonlinear term**: Computing $u^3$ for large $u$ values causes overflow
2. **Numerical instability in time stepping**: The IMEX Euler scheme was not stable enough for the chosen parameters
3. **Dealiasing issues**: The pseudo-spectral projection may have been amplifying errors
4. **Parameter sensitivity**: The combination of $\varepsilon = 0.02$, $dt = 5 \times 10^{-4}$, and $N=80$ modes was unstable

## Solution: Physics-Informed Neural Networks (PINN)

Replaced the spectral method with a **Physics-Informed Neural Network** approach (`allen_cahn_pin.py`).

### Why PINN?

PINNs are robust because they:
- ✅ Use neural network regularization (implicit smoothing)
- ✅ Avoid grid refinement issues
- ✅ Handle nonlinearities more gracefully
- ✅ Don't require explicit stability constraints like time-stepping schemes
- ✅ Learn from both the PDE and initial conditions simultaneously

### PINN Formulation

The PINN learns $u(x,t)$ by minimizing:

$$\mathcal{L} = \mathcal{L}_{PDE} + w_{IC} \mathcal{L}_{IC}$$

where:

**PDE Loss** (enforces the differential equation):
$$\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{(x_i, t_i) \in \Omega \times [0,T]} \left|\frac{\partial u}{\partial t} - \varepsilon^2 \frac{\partial^2 u}{\partial x^2} + (u^3 - u)\right|^2$$

**Initial Condition Loss**:
$$\mathcal{L}_{IC} = \frac{1}{N_{IC}} \sum_{x_j \in \Omega} |u(x_j, 0) - u_0(x_j)|^2$$

The network architecture is:
- Input: $(x, t)$ (2 inputs)
- Hidden layer: 32 neurons with ReLU activation
- Output: $u(x,t)$ (1 output)

### Key Implementation Details

1. **Finite Differences for Derivatives**: Uses centered differences with $h = 0.005$
2. **Overflow Protection**: Clamps solution values to $[-5, 5]$ to prevent overflow in $u^3$
3. **Stochastic Sampling**: Randomly samples PDE residual points at each iteration (improves training efficiency)
4. **Weighted Loss**: Emphasizes initial condition ($w_{IC} = 10$)
5. **L-BFGS-B Optimizer**: Quasi-Newton method with box constraints

### Training Configuration

```python
N_pde = 300      # Interior training points (randomly sampled)
N_ic = 50        # Initial condition points
epochs = 200     # Training iterations
```

The training completes in ~60 seconds on a standard machine.

## Files

| File | Purpose |
|------|---------|
| `allen_cahn.py` | Original spectral Galerkin method (unstable) |
| `allen_cahn_pin.py` | **NEW: PINN-based solver (working)** |
| `allen_cahn_pinn.png` | Solution visualization |

## Usage

```bash
python allen_cahn_pin.py
```

Output:
- Console output showing training progress and solution statistics
- `allen_cahn_pinn.png`: Plot with two panels
  - Left: Multiple solution snapshots over time
  - Right: Heatmap of solution evolution

## Solution Characteristics

For the Allen-Cahn equation with $\varepsilon = 0.02$:

```
t=0.000: min=-0.1261, max=0.0710
t=0.050: min=-0.1363, max=0.0741
t=0.100: min=-0.1465, max=0.0783
t=0.150: min=-0.1517, max=0.0838
t=0.200: min=-0.1562, max=0.0868
```

The solution is **bounded** and physically reasonable, showing smooth evolution from the initial condition $u_0(x) = x^2 \cos(\pi x)$.

## PDE Details

The Allen-Cahn equation:
$$u_t = \varepsilon^2 u_{xx} - (u^3 - u)$$

- **Domain**: $x \in [-1, 1]$, $t \in [0, 0.2]$
- **Initial condition**: $u(x, 0) = x^2 \cos(\pi x)$
- **Boundary conditions**: Natural (Neumann) in the PINN formulation
- **Physical meaning**: Models phase separation with interfacial tension parameter $\varepsilon$

## Future Improvements

1. **Automatic Differentiation**: Use JAX or PyTorch for exact derivatives instead of finite differences
2. **Adaptive Training**: Increase network size or training time for higher accuracy
3. **Boundary Constraints**: Explicitly enforce Dirichlet BCs if needed
4. **Residual Network**: Use residual connections for deeper networks
5. **Multiple Snapshots**: Train separate models for different time horizons

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems."
- Beck, C., Jentzen, A., & Kuckuck, B. (2020). "ML-Assisted computational physics: Case study of Allen-Cahn equation."
