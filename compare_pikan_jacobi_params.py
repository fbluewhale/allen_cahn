"""
Comparison of PIKAN-Jacobi with different parameter configurations.

This script trains multiple PIKAN-Jacobi models, each with a different set of
Jacobi polynomial activation parameters (order, alpha, beta), and compares
their performance in solving the Allen-Cahn PDE.

PDE:  u_t = ε² u_xx − (u³ − u)       Allen-Cahn equation
Domain:  x ∈ [−1, 1],  t ∈ [0, 0.2]

Dependencies: numpy, scipy, matplotlib, autograd
"""

import time
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Use standard numpy for non-differentiable parts
import numpy as onp


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def u0_func(x):
    return x ** 2 * onp.cos(onp.pi * x)


def compute_residual(forward_fn, params, x, t, eps, h=0.002):
    """Compute the PDE residual at given points (x, t)."""
    u    = forward_fn(params, x, t)
    u_tp = forward_fn(params, x, np.minimum(t + h, 1.0))
    u_tm = forward_fn(params, x, np.maximum(t - h, 0.0))
    u_t  = (u_tp - u_tm) / (2 * h)

    u_xp = forward_fn(params, np.minimum(x + h, 1.0), t)
    u_xm = forward_fn(params, np.maximum(x - h, -1.0), t)
    u_xx = (u_xp - 2 * u + u_xm) / h ** 2

    res = u_t - eps ** 2 * u_xx + u ** 3 - u
    return res

# ═══════════════════════════════════════════════════════════════════════════════
#  Pure-function neural networks compatible with autograd
#  (autograd needs pure functions — no mutable class state during diff)
# ═══════════════════════════════════════════════════════════════════════════════

# ── PIKAN-Jacobi (learnable Jacobi polynomial activations) ───────────────────

def pikanj_init(layer_sizes, order=6, alpha=-0.5, beta=-0.5, seed=42):
    """Return (params_list, meta_dict)."""
    rng = onp.random.RandomState(seed)
    params = []
    # weights & biases
    for i in range(len(layer_sizes) - 1):
        lim = onp.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
        params.append(rng.uniform(-lim, lim, (layer_sizes[i], layer_sizes[i+1])))
        params.append(onp.zeros(layer_sizes[i+1]))
    # activation coefficients for hidden layers
    n_hidden_layers = len(layer_sizes) - 2
    for li in range(n_hidden_layers):
        n_neurons = layer_sizes[li + 1]
        # shape (n_neurons, order) — each row is coefficients for one neuron
        c = onp.zeros((n_neurons, order))
        c[:, min(1, order-1)] = 1.0  # initialise near identity
        params.append(c)
    meta = {"layer_sizes": layer_sizes, "order": order,
            "alpha": alpha, "beta": beta, "n_hidden": n_hidden_layers}
    return params, meta

def _jacobi_basis_batch(z, order, alpha, beta):
    """Evaluate Jacobi P_0..P_{order-1} on z of shape (N, M).
    Returns (N, M, order) — fully vectorized, no per-neuron loop."""
    basis = [np.ones_like(z)]              # P_0 = 1
    if order > 1:
        basis.append(0.5 * ((alpha - beta) + (alpha + beta + 2.0) * z))
    for n in range(2, order):
        a1 = 2.0*n*(n + alpha + beta)*(2*n + alpha + beta - 2)
        a2 = (2*n + alpha + beta - 1)*((2*n + alpha + beta)*(2*n + alpha + beta - 2)*z
              + alpha**2 - beta**2)
        a3 = 2.0*(n + alpha - 1)*(n + beta - 1)*(2*n + alpha + beta)
        basis.append((a2 * basis[-1] - a3 * basis[-2]) / (a1 + 1e-30))
    return np.stack(basis, axis=-1)        # (N, M, order)

def pikanj_forward(params, meta, x, t):
    """Forward pass for PIKAN-Jacobi — fully vectorized."""
    n_layers = (len(params) - meta["n_hidden"]) // 2
    z = np.column_stack([x, t])
    wb_params = params[:2*n_layers]
    act_params = params[2*n_layers:]

    for i in range(n_layers - 1):
        W, b = wb_params[2*i], wb_params[2*i+1]
        z = z @ W + b                              # (N, neurons)
        c = act_params[i]                           # (neurons, order)
        z_clip = np.clip(z, -1.0, 1.0)
        # batch Jacobi: (N, neurons, order) @ (neurons, order, 1) → (N, neurons, 1)
        B = _jacobi_basis_batch(z_clip, meta["order"],
                                meta["alpha"], meta["beta"])  # (N, neurons, order)
        # einsum: sum over order dimension for each neuron independently
        z = np.einsum('nmo,mo->nm', B, c)           # (N, neurons)
    W, b = wb_params[-2], wb_params[-1]
    return (z @ W + b).ravel()


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic PDE loss + training (works with any forward function)
# ═══════════════════════════════════════════════════════════════════════════════

def make_loss_fn(forward_fn, eps, h=0.002):
    """Return a loss(params, xp, tp, xi, ti, ui) function."""

    def loss(params, xp, tp, xi, ti, ui):
        # PDE residual
        u    = forward_fn(params, xp, tp)
        u_tp = forward_fn(params, xp, np.minimum(tp + h, 1.0))
        u_tm = forward_fn(params, xp, np.maximum(tp - h, 0.0))
        u_t  = (u_tp - u_tm) / (2 * h)

        u_xp = forward_fn(params, np.minimum(xp + h, 1.0), tp)
        u_xm = forward_fn(params, np.maximum(xp - h, -1.0), tp)
        u_xx = (u_xp - 2 * u + u_xm) / h ** 2

        res = u_t - eps ** 2 * u_xx + u ** 3 - u
        l_pde = np.mean(res ** 2)

        # IC loss
        u_ic = forward_fn(params, xi, ti)
        l_ic = np.mean((u_ic - ui) ** 2)

        return l_pde + 20.0 * l_ic

    return loss


def flatten(params):
    """Flatten list of arrays into 1-D vector."""
    return onp.concatenate([p.ravel() for p in params])

def unflatten(vec, shapes):
    """Reconstruct list of arrays from 1-D vector."""
    out, idx = [], 0
    for s in shapes:
        n = 1
        for d in s: n *= d
        out.append(vec[idx:idx+n].reshape(s))
        idx += n
    return out


def train_adam_lbfgs(loss_fn, params, xp, tp, xi, ti, ui,
                     adam_steps=1500, adam_lr=1e-3, lbfgs_steps=300, verbose=True):
    """Two-stage training: Adam (exploration) → L-BFGS-B (refinement).
    loss_fn must accept (params_list, xp, tp, xi, ti, ui)."""

    shapes = [p.shape for p in params]
    grad_fn = grad(loss_fn, argnum=0)

    # ── Adam ──────────────────────────────────────────────────────────────
    if verbose: print("    Adam stage …")
    m = [onp.zeros_like(p) for p in params]
    v = [onp.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for step in range(1, adam_steps + 1):
        # cosine LR decay
        lr = adam_lr * 0.5 * (1 + onp.cos(onp.pi * step / adam_steps))

        g = grad_fn(params, xp, tp, xi, ti, ui)
        for j in range(len(params)):
            gj = onp.array(g[j])
            m[j] = beta1 * m[j] + (1 - beta1) * gj
            v[j] = beta2 * v[j] + (1 - beta2) * gj ** 2
            m_hat = m[j] / (1 - beta1 ** step)
            v_hat = v[j] / (1 - beta2 ** step)
            params[j] = onp.array(params[j]) - lr * m_hat / (onp.sqrt(v_hat) + eps_adam)

        if verbose and (step % 300 == 0 or step == 1):
            l = float(loss_fn(params, xp, tp, xi, ti, ui))
            print(f"      step {step:5d}/{adam_steps}  lr={lr:.2e}  loss={l:.6e}")

    # ── L-BFGS-B with analytical gradients ────────────────────────────────
    if verbose: print("    L-BFGS-B refinement …")
    cc = [0]

    def objective(vec):
        p = unflatten(vec, shapes)
        return float(loss_fn(p, xp, tp, xi, ti, ui))

    def gradient(vec):
        p = unflatten(vec, shapes)
        g = grad_fn(p, xp, tp, xi, ti, ui)
        return onp.array(flatten(g), dtype=float)

    def cb(vec):
        cc[0] += 1
        if verbose and cc[0] % 50 == 0:
            l = objective(vec)
            print(f"      L-BFGS iter {cc[0]:3d}: loss={l:.6e}")

    x0 = flatten(params)
    res = minimize(objective, x0, jac=gradient, method="L-BFGS-B",
                   callback=cb if verbose else None,
                   options={"maxiter": lbfgs_steps, "ftol": 1e-10,
                            "maxfun": lbfgs_steps * 3})
    params = unflatten(res.x, shapes)
    final_loss = float(loss_fn(params, xp, tp, xi, ti, ui))
    return params, final_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    eps_val = 0.01
    T = 0.2
    N_pde = 600
    N_ic  = 150
    adam_steps = 1500
    lbfgs_steps = 300
    n_snap = 5

    print("=" * 76)
    print("  Comparison of PIKAN-Jacobi with different parameter configurations")
    print("=" * 76)
    print(f"  PDE : u_t = ε² u_xx − (u³ − u),  ε = {eps_val}")
    print(f"  Training : {N_pde} PDE + {N_ic} IC points")
    print(f"  Optimizer: Adam ({adam_steps} steps) → L-BFGS-B ({lbfgs_steps} steps)")
    print(f"  Key : analytical gradients via autograd")
    print("=" * 76, "\n")

    # Define different Jacobi parameter configurations to test
    jacobi_configs = {
        "PJ-Opt":   {"order": 6, "alpha": -0.5, "beta": -0.5}, # Optimal from previous sweep
        "PJ-Leg":   {"order": 6, "alpha": 0.0,  "beta": 0.0},  # Legendre Polynomials
        "PJ-Cheb1": {"order": 6, "alpha": -0.5, "beta": -0.5}, # Chebyshev of 1st kind (alias for PJ-Opt)
        "PJ-Cheb2": {"order": 6, "alpha": 0.5,  "beta": 0.5},  # Chebyshev of 2nd kind
        "PJ-LowO":  {"order": 4, "alpha": -0.5, "beta": -0.5}, # Lower order for comparison
    }

    # ── training data ─────────────────────────────────────────────────────
    rng = onp.random.RandomState(0)
    xp = rng.uniform(-1, 1, N_pde).astype(float)
    tp = rng.uniform(0.001, T, N_pde).astype(float)
    xi = onp.linspace(-1, 1, N_ic).astype(float)
    ti = onp.zeros(N_ic)
    ui = u0_func(xi)

    # ── evaluation grid (residual only) ───────────────────────────────────
    xs = onp.linspace(-1, 1, 100)
    snap_t = onp.linspace(0, T, n_snap)

    # ── run PIKAN-Jacobi for each configuration ───────────────────────────
    solvers = {}
    for i, (name, config) in enumerate(jacobi_configs.items()):
        print(f"[{i+1}/{len(jacobi_configs)}] {name} (order={config['order']}, α={config['alpha']}, β={config['beta']}) …")
        pj_params, pj_meta = pikanj_init([2, 20, 20, 1], **config)
        pj_loss_fn = make_loss_fn(
            lambda params, x, t: pikanj_forward(params, pj_meta, x, t), eps_val)
        t0 = time.time()
        pj_params, pj_loss = train_adam_lbfgs(
            pj_loss_fn, pj_params, xp, tp, xi, ti, ui,
            adam_steps=adam_steps, lbfgs_steps=lbfgs_steps)
        pj_time = time.time() - t0
        pj_nparams = sum(p.size for p in pj_params)
        pj_res_snaps = [onp.array(
            compute_residual(
                lambda params, x, t: pikanj_forward(params, pj_meta, x, t),
                pj_params,
                xs,
                onp.full_like(xs, tv),
                eps_val,
            )
        ) for tv in snap_t]
        pj_res_l2 = onp.sqrt(onp.mean(onp.concatenate([s.ravel()**2 for s in pj_res_snaps])))
        print(f"  loss={pj_loss:.6e}  ResL2={pj_res_l2:.6e}  time={pj_time:.1f}s\n")
        solvers[name] = {"loss": pj_loss, "time": pj_time,
                         "nparams": pj_nparams, "res_snaps": pj_res_snaps,
                         "res_l2": pj_res_l2}

    # ── results table ─────────────────────────────────────────────────────
    names = list(solvers.keys())
    print("=" * 76)
    print(f"{'Metric':<28}", end="")
    for n in names: print(f"{n:>16}", end="")
    print("\n" + "-" * 76)
    for label, key, fmt in [
        ("# Parameters", "nparams", ",d"),
        ("Final loss", "loss", ".4e"),
        ("L₂ Residual Error", "res_l2", ".4e"),
        ("Training time (s)", "time", ".1f"),
    ]:
        print(f"{label:<28}", end="")
        for n in names:
            v = solvers[n][key]
            print(f"{v:>16{fmt}}", end="")
        print()
    print("=" * 76)

    best_time = min(names, key=lambda n: solvers[n]["time"])
    best_res_l2 = min(names, key=lambda n: solvers[n]["res_l2"])
    print(f"\n  ✓ Most accurate (L2 residual) → {best_res_l2}")
    print(f"  ✓ Fastest                    → {best_time}\n")

    # ── residual plots only ───────────────────────────────────────────────
    plot_data_res = {n: solvers[n]["res_snaps"] for n in names}
    ncols = len(names)

    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))
    for col, pname in enumerate(names):
        ax = axes[0, col]
        for i, tv in enumerate(snap_t):
            ax.plot(xs, onp.abs(plot_data_res[pname][i]), lw=2, label=f"t={tv:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("|Residual|"); ax.set_ylim(0, 0.1)
        ax.set_title(pname, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        mat = onp.array(plot_data_res[pname])
        im = ax.contourf(xs, snap_t, mat, levels=20, cmap="viridis")
        plt.colorbar(im, ax=ax, label="|Residual|")
        ax.set_xlabel("x"); ax.set_ylabel("t")
        ax.set_title(f"{pname} residual heatmap", fontsize=12)

    fig.suptitle("PIKAN-Jacobi Parameter Comparison: Residual Error",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("compare_pikan_jacobi_residual.png", dpi=150, bbox_inches="tight")
    print("Saved → compare_pikan_jacobi_residual.png\n")
    plt.close()


if __name__ == "__main__":
    main()