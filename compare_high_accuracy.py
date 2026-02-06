"""
High-Accuracy Comparison:  PINN  vs  PIKAN-Spline  vs  PIKAN-Jacobi
=====================================================================
Key improvements over the previous compare_3way.py:

  1. **Analytical gradients** via `autograd` — eliminates the massive
     overhead of L-BFGS-B's numerical gradient estimation (was 2×n_params
     loss evals per step — the #1 reason accuracy was terrible).
  2. **Adam + L-BFGS-B** two-stage optimizer — Adam for global exploration,
     then L-BFGS-B with exact gradients for fine-tuning.
  3. **Learning-rate schedule** — cosine decay for Adam.
  4. **Deeper PINN** — 2 hidden layers instead of 1, with tanh activation
     (tanh works much better than Jacobi P₃ for a fixed activation).
  5. Larger training set, more iterations.

PDE:  u_t = ε² u_xx − (u³ − u)       Allen-Cahn equation
Domain:  x ∈ [−1, 1],  t ∈ [0, 0.2]

Dependencies: numpy, scipy, matplotlib, autograd
"""

import time
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
from scipy.special import eval_jacobi
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

def fd_reference(eps, T, nx=200, nt=4000):
    x = onp.linspace(-1, 1, nx)
    t = onp.linspace(0, T, nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    r = eps ** 2 * dt / dx ** 2
    print(f"  FD grid {nx}×{nt}, CFL r={r:.4f}")
    u = onp.zeros((nt, nx)); u[0] = u0_func(x)
    for n in range(nt - 1):
        uc = u[n]
        uxx = onp.zeros_like(uc)
        uxx[1:-1] = (uc[2:] - 2 * uc[1:-1] + uc[:-2]) / dx ** 2
        uxx[0], uxx[-1] = uxx[1], uxx[-2]
        u[n + 1] = onp.clip(uc + dt * (eps ** 2 * uxx - (uc ** 3 - uc)), -2, 2)
    return x, t, u


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure-function neural networks compatible with autograd
#  (autograd needs pure functions — no mutable class state during diff)
# ═══════════════════════════════════════════════════════════════════════════════

# ── PINN (tanh, 2 hidden layers) ─────────────────────────────────────────────

def pinn_init(layer_sizes, seed=42):
    """Return list of (W, b) pairs."""
    rng = onp.random.RandomState(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        lim = onp.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
        W = rng.uniform(-lim, lim, (layer_sizes[i], layer_sizes[i+1]))
        b = onp.zeros(layer_sizes[i+1])
        params.append(W)
        params.append(b)
    return params

def pinn_forward(params, x, t):
    """Forward pass: x, t are 1-D arrays → returns 1-D array."""
    z = np.column_stack([x, t])                  # (N, 2)
    n_layers = len(params) // 2
    for i in range(n_layers - 1):
        W, b = params[2*i], params[2*i+1]
        z = np.tanh(z @ W + b)                    # hidden layers
    W, b = params[-2], params[-1]
    return (z @ W + b).ravel()                     # output layer (linear)


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


# ── PIKAN-Spline ─────────────────────────────────────────────────────────────

def pikans_init(layer_sizes, n_knots=9, seed=42):
    """Spline KAN — knot values are the learnable parameters."""
    rng = onp.random.RandomState(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        lim = onp.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
        params.append(rng.uniform(-lim, lim, (layer_sizes[i], layer_sizes[i+1])))
        params.append(onp.zeros(layer_sizes[i+1]))
    n_hidden = len(layer_sizes) - 2
    knot_pos = onp.linspace(-1, 1, n_knots)
    for li in range(n_hidden):
        n_neurons = layer_sizes[li + 1]
        params.append(rng.randn(n_neurons, n_knots) * 0.1)
    meta = {"layer_sizes": layer_sizes, "n_knots": n_knots,
            "knot_pos": knot_pos, "n_hidden": n_hidden}
    return params, meta

def _spline_eval_batch(z, knot_pos, knot_vals):
    """Piecewise-linear interpolation for z of shape (N, M),
    knot_vals of shape (M, K). Returns (N, M). Vectorized via flat indexing."""
    K = len(knot_pos)
    N, M = z.shape
    z_clip = np.clip(z, knot_pos[0], knot_pos[-1])
    dx = knot_pos[1] - knot_pos[0]
    idx_f = (z_clip - knot_pos[0]) / dx              # (N, M)
    idx = np.clip(np.floor(idx_f).astype(int), 0, K - 2)  # (N, M)
    frac = np.clip(idx_f - idx, 0.0, 1.0)            # (N, M)
    # Flatten knot_vals (M, K) → (M*K,)
    # For neuron m, index i → flat index m*K + i
    kv_flat = knot_vals.ravel()                       # (M*K,)
    offsets = (np.arange(M) * K).reshape(1, M)        # (1, M) broadcast
    flat_idx = offsets + idx                           # (N, M)
    v0 = kv_flat[flat_idx]                            # (N, M)
    v1 = kv_flat[flat_idx + 1]                        # (N, M)
    return v0 * (1.0 - frac) + v1 * frac
    return v0 * (1.0 - frac) + v1 * frac

def pikans_forward(params, meta, x, t):
    """Forward pass for PIKAN-Spline — vectorized."""
    n_layers = (len(params) - meta["n_hidden"]) // 2
    z = np.column_stack([x, t])
    wb_params = params[:2*n_layers]
    act_params = params[2*n_layers:]
    kp = meta["knot_pos"]

    for i in range(n_layers - 1):
        W, b = wb_params[2*i], wb_params[2*i+1]
        z = z @ W + b
        kv = act_params[i]  # (neurons, n_knots)
        z_clip = np.clip(z, -1.0, 1.0)
        z = _spline_eval_batch(z_clip, kp, kv)
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
                     adam_steps=1500, adam_lr=1e-3, lbfgs_steps=300):
    """Two-stage training: Adam (exploration) → L-BFGS-B (refinement).
    loss_fn must accept (params_list, xp, tp, xi, ti, ui)."""

    shapes = [p.shape for p in params]
    grad_fn = grad(loss_fn, argnum=0)

    # ── Adam ──────────────────────────────────────────────────────────────
    print("    Adam stage …")
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

        if step % 300 == 0 or step == 1:
            l = float(loss_fn(params, xp, tp, xi, ti, ui))
            print(f"      step {step:5d}/{adam_steps}  lr={lr:.2e}  loss={l:.6e}")

    # ── L-BFGS-B with analytical gradients ────────────────────────────────
    print("    L-BFGS-B refinement …")
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
        if cc[0] % 50 == 0:
            l = objective(vec)
            print(f"      L-BFGS iter {cc[0]:3d}: loss={l:.6e}")

    x0 = flatten(params)
    res = minimize(objective, x0, jac=gradient, method="L-BFGS-B",
                   callback=cb,
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
    print("  High-Accuracy:  PINN  vs  PIKAN-Spline  vs  PIKAN-Jacobi")
    print("=" * 76)
    print(f"  PDE : u_t = ε² u_xx − (u³ − u),  ε = {eps_val}")
    print(f"  Training : {N_pde} PDE + {N_ic} IC points")
    print(f"  Optimizer: Adam ({adam_steps} steps) → L-BFGS-B ({lbfgs_steps} steps)")
    print(f"  Key : analytical gradients via autograd")
    print("=" * 76, "\n")

    # ── FD reference ──────────────────────────────────────────────────────
    print("[0] FD reference …")
    t0 = time.time()
    x_fd, t_fd, u_fd = fd_reference(eps_val, T)
    print(f"  done {time.time()-t0:.2f}s\n")

    # ── training data ─────────────────────────────────────────────────────
    rng = onp.random.RandomState(0)
    xp = rng.uniform(-1, 1, N_pde).astype(float)
    tp = rng.uniform(0.001, T, N_pde).astype(float)
    xi = onp.linspace(-1, 1, N_ic).astype(float)
    ti = onp.zeros(N_ic)
    ui = u0_func(xi)

    # ── evaluation grid ───────────────────────────────────────────────────
    xs = onp.linspace(-1, 1, 100)
    snap_t = onp.linspace(0, T, n_snap)
    fd_snaps = []
    for tv in snap_t:
        idx = onp.argmin(onp.abs(t_fd - tv))
        fd_snaps.append(onp.interp(xs, x_fd, u_fd[idx]))

    def rel_l2(pred, ref):
        num = sum(onp.sum((p - r) ** 2) for p, r in zip(pred, ref))
        den = sum(onp.sum(r ** 2) for r in ref) + 1e-12
        return onp.sqrt(num / den)

    # ── define solvers ────────────────────────────────────────────────────
    solvers = {}

    # 1) PINN — tanh, 2 hidden layers
    print("[1/3] PINN  (2,64,64,1)  tanh …")
    pinn_params = pinn_init([2, 64, 64, 1])
    pinn_loss_fn = make_loss_fn(
        lambda params, x, t: pinn_forward(params, x, t), eps_val)
    t0 = time.time()
    pinn_params, pinn_loss = train_adam_lbfgs(
        pinn_loss_fn, pinn_params, xp, tp, xi, ti, ui,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps)
    pinn_time = time.time() - t0
    pinn_nparams = sum(p.size for p in pinn_params)
    pinn_snaps = [onp.array(pinn_forward(pinn_params, xs, onp.full_like(xs, tv)))
                  for tv in snap_t]
    pinn_l2 = rel_l2(pinn_snaps, fd_snaps)
    pinn_res_snaps = [onp.array(compute_residual(pinn_forward, pinn_params, xs, onp.full_like(xs, tv), eps_val))
                      for tv in snap_t]
    pinn_res_l2 = onp.sqrt(onp.mean(onp.concatenate([s.ravel()**2 for s in pinn_res_snaps])))
    print(f"  loss={pinn_loss:.6e}  L2={pinn_l2:.6e}  ResL2={pinn_res_l2:.6e}  time={pinn_time:.1f}s\n")
    solvers["PINN"] = {"loss": pinn_loss, "l2": pinn_l2, "time": pinn_time,
                       "nparams": pinn_nparams, "snaps": pinn_snaps, "res_snaps": pinn_res_snaps, "res_l2": pinn_res_l2}

    # 2) PIKAN-Jacobi — order=6, α=−0.5, β=−0.5
    print("[2/3] PIKAN-Jacobi  (2,20,20,1)  order=6  α=-0.5  β=-0.5 …")
    pj_params, pj_meta = pikanj_init([2, 20, 20, 1], order=6,
                                     alpha=-0.5, beta=-0.5)
    pj_loss_fn = make_loss_fn(
        lambda params, x, t: pikanj_forward(params, pj_meta, x, t), eps_val)
    t0 = time.time()
    pj_params, pj_loss = train_adam_lbfgs(
        pj_loss_fn, pj_params, xp, tp, xi, ti, ui,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps)
    pj_time = time.time() - t0
    pj_nparams = sum(p.size for p in pj_params)
    pj_snaps = [onp.array(pikanj_forward(pj_params, pj_meta, xs,
                                         onp.full_like(xs, tv)))
                for tv in snap_t]
    pj_l2 = rel_l2(pj_snaps, fd_snaps)
    pj_res_snaps = [onp.array(compute_residual(lambda params, x, t: pikanj_forward(params, pj_meta, x, t), pj_params, xs, onp.full_like(xs, tv), eps_val))
                    for tv in snap_t]
    pj_res_l2 = onp.sqrt(onp.mean(onp.concatenate([s.ravel()**2 for s in pj_res_snaps])))
    print(f"  loss={pj_loss:.6e}  L2={pj_l2:.6e}  ResL2={pj_res_l2:.6e}  time={pj_time:.1f}s\n")
    solvers["PIKAN-Jacobi"] = {"loss": pj_loss, "l2": pj_l2, "time": pj_time,
                               "nparams": pj_nparams, "snaps": pj_snaps, "res_snaps": pj_res_snaps, "res_l2": pj_res_l2}

    # 3) PIKAN-Spline — 9-knot piecewise-linear (autograd-friendly)
    print("[3/3] PIKAN-Spline  (2,20,20,1)  9 knots …")
    ps_params, ps_meta = pikans_init([2, 20, 20, 1], n_knots=9)
    ps_loss_fn = make_loss_fn(
        lambda params, x, t: pikans_forward(params, ps_meta, x, t), eps_val)
    t0 = time.time()
    ps_params, ps_loss = train_adam_lbfgs(
        ps_loss_fn, ps_params, xp, tp, xi, ti, ui,
        adam_steps=adam_steps, lbfgs_steps=lbfgs_steps)
    ps_time = time.time() - t0
    ps_nparams = sum(p.size for p in ps_params)
    ps_snaps = [onp.array(pikans_forward(ps_params, ps_meta, xs,
                                         onp.full_like(xs, tv)))
                for tv in snap_t]
    ps_l2 = rel_l2(ps_snaps, fd_snaps)
    ps_res_snaps = [onp.array(compute_residual(lambda params, x, t: pikans_forward(params, ps_meta, x, t), ps_params, xs, onp.full_like(xs, tv), eps_val))
                    for tv in snap_t]
    ps_res_l2 = onp.sqrt(onp.mean(onp.concatenate([s.ravel()**2 for s in ps_res_snaps])))
    print(f"  loss={ps_loss:.6e}  L2={ps_l2:.6e}  ResL2={ps_res_l2:.6e}  time={ps_time:.1f}s\n")
    solvers["PIKAN-Spline"] = {"loss": ps_loss, "l2": ps_l2, "time": ps_time,
                               "nparams": ps_nparams, "snaps": ps_snaps, "res_snaps": ps_res_snaps, "res_l2": ps_res_l2}

    # ── results table ─────────────────────────────────────────────────────
    names = list(solvers.keys())
    print("=" * 76)
    print(f"{'Metric':<28}", end="")
    for n in names: print(f"{n:>16}", end="")
    print("\n" + "-" * 76)
    for label, key, fmt in [
        ("# Parameters", "nparams", ",d"),
        ("Final loss", "loss", ".4e"),
        ("Rel. L₂ error vs FD", "l2", ".4e"),
        ("L₂ Residual Error", "res_l2", ".4e"),
        ("Training time (s)", "time", ".1f"),
    ]:
        print(f"{label:<28}", end="")
        for n in names:
            v = solvers[n][key]
            print(f"{v:>16{fmt}}", end="")
        print()
    print("=" * 76)

    best_l2 = min(names, key=lambda n: solvers[n]["l2"])
    best_time = min(names, key=lambda n: solvers[n]["time"])
    print(f"\n  ✓ Most accurate (L2 sol) → {best_l2}")
    print(f"  ✓ Fastest                → {best_time}\n")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_data = {"FD Ref": fd_snaps}
    for n in names: plot_data[n] = solvers[n]["snaps"]
    plot_names = ["FD Ref"] + names
    ncols = len(plot_names)

    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 15))
    for col, pname in enumerate(plot_names):
        # Solution plots
        ax = axes[0, col]
        for i, tv in enumerate(snap_t):
            ax.plot(xs, plot_data[pname][i], lw=2, label=f"t={tv:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_ylim(-1.5, 1.5)
        ax.set_title(pname, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Heatmap of solution
        ax = axes[1, col]
        mat = onp.array(plot_data[pname])
        im = ax.contourf(xs, snap_t, mat, levels=20, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("x"); ax.set_ylabel("t")
        ax.set_title(f"{pname} heatmap", fontsize=12)
    
        # Residual plots (only for models, not FD Ref)
        if pname != "FD Ref":
            ax = axes[2, col]
            for i, tv in enumerate(snap_t):
                ax.plot(xs, onp.abs(solvers[pname]["res_snaps"][i]), lw=1.5, label=f"t={tv:.3f}")
            ax.set_xlabel("x"); ax.set_ylabel("|Residual|")
            ax.set_title(f"{pname} residual", fontweight="bold")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        else:
            # For FD Ref, plot zeros for residual or leave blank
            ax = axes[2, col]
            ax.plot(xs, onp.zeros_like(xs), lw=1.5, ls='--', color='gray', label='Ideal Residual = 0')
            ax.set_xlabel("x"); ax.set_ylabel("|Residual|")
            ax.set_title("FD Ref residual", fontweight="bold")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("High-Accuracy:  PINN vs PIKAN-Spline vs PIKAN-Jacobi",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("compare_high_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved → compare_high_accuracy.png")
    plt.close()

    # error curves (solution error)
    fig2, axes2 = plt.subplots(1, len(names), figsize=(6*len(names), 5))
    for col, name in enumerate(names):
        ax = axes2[col]
        for i, tv in enumerate(snap_t):
            ax.plot(xs, onp.abs(
                onp.array(solvers[name]["snaps"][i]) - fd_snaps[i]),
                lw=1.5, label=f"t={tv:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("|error|")
        ax.set_title(f"{name} solution error vs FD", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_high_accuracy_solution_errors.png", dpi=150, bbox_inches="tight")
    print("Saved → compare_high_accuracy_solution_errors.png\n")
    plt.close()

    # error curves (residual error)
    fig3, axes3 = plt.subplots(1, len(names), figsize=(6*len(names), 5))
    for col, name in enumerate(names):
        ax = axes3[col]
        for i, tv in enumerate(snap_t):
            ax.plot(xs, onp.abs(solvers[name]["res_snaps"][i]),
                lw=1.5, label=f"t={tv:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("|Residual|")
        ax.set_title(f"{name} residual error", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compare_high_accuracy_residual_errors.png", dpi=150, bbox_inches="tight")
    print("Saved → compare_high_accuracy_residual_errors.png\n")
    plt.close()


if __name__ == "__main__":
    main()
