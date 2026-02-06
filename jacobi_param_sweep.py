"""
PIKAN-Jacobi Parameter Sweep
==============================
Tests different Jacobi polynomial parameters (order, α, β) for the
PIKAN-Jacobi solver on the Allen-Cahn equation, to find the optimal
activation function configuration.

Sweep grid:
  • order  ∈ {3, 4, 5, 6, 8}
  • α      ∈ {-0.5, 0.0, 0.5, 1.0}
  • β      ∈ {-0.5, 0.0, 0.5, 1.0}

Total: 5 × 4 × 4 = 80 runs  (each ~15-25 s → ~25 min total)
Uses reduced settings for speed; final winner is re-trained with full budget.

Dependencies: numpy, scipy, matplotlib
"""

import time
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_jacobi
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared
# ═══════════════════════════════════════════════════════════════════════════════

def u0_func(x):
    return x ** 2 * np.cos(np.pi * x)


def fd_reference(eps, T, nx=200, nt=4000):
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, T, nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    u = np.zeros((nt, nx)); u[0] = u0_func(x)
    for n in range(nt - 1):
        uc = u[n]
        uxx = np.zeros_like(uc)
        uxx[1:-1] = (uc[2:] - 2 * uc[1:-1] + uc[:-2]) / dx ** 2
        uxx[0], uxx[-1] = uxx[1], uxx[-2]
        u[n + 1] = np.clip(uc + dt * (eps ** 2 * uxx - (uc ** 3 - uc)), -2, 2)
    return x, t, u


# ═══════════════════════════════════════════════════════════════════════════════
#  PIKAN-Jacobi solver  (compact, self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

class _JacobiAct:
    def __init__(self, order, alpha, beta):
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.c = np.zeros(order)
        self.c[min(1, order - 1)] = 1.0

    def __call__(self, x):
        xc = np.clip(np.asarray(x, float), -1, 1)
        out = np.zeros_like(xc)
        for k in range(self.order):
            out += self.c[k] * eval_jacobi(k, self.alpha, self.beta, xc)
        return out

    def get(self): return self.c.copy()
    def set(self, v): self.c = v.copy()


class PIKANJacobi:
    def __init__(self, eps, n_hidden=16, order=6, alpha=0.0, beta=0.0):
        self.eps = eps
        self.layers = (2, n_hidden, n_hidden, 1)
        self.order = order
        self.alpha = alpha
        self.beta = beta
        np.random.seed(42)
        self.W, self.b, self.acts = [], [], []
        for i in range(len(self.layers) - 1):
            lim = np.sqrt(6.0 / (self.layers[i] + self.layers[i + 1]))
            self.W.append(np.random.uniform(-lim, lim,
                                            (self.layers[i], self.layers[i + 1])))
            self.b.append(np.zeros((1, self.layers[i + 1])))
        for i in range(len(self.layers) - 2):
            self.acts.append([_JacobiAct(order, alpha, beta)
                              for _ in range(self.layers[i + 1])])

    def _pack(self):
        parts = [w.ravel() for w in self.W] + [b.ravel() for b in self.b]
        for la in self.acts:
            for a in la: parts.append(a.get())
        return np.concatenate(parts)

    def _unpack(self, p):
        idx = 0
        for i, w in enumerate(self.W):
            s = w.size; self.W[i] = p[idx:idx + s].reshape(w.shape); idx += s
        for i, b in enumerate(self.b):
            s = b.size; self.b[i] = p[idx:idx + s].reshape(b.shape); idx += s
        for la in self.acts:
            for a in la: a.set(p[idx:idx + self.order]); idx += self.order

    def forward(self, x, t):
        z = np.column_stack([np.clip(x, -1, 1), np.clip(t, 0, 1)])
        for li in range(len(self.W) - 1):
            z = z @ self.W[li] + self.b[li]
            for ni in range(z.shape[1]):
                z[:, ni] = self.acts[li][ni](z[:, ni])
        return (z @ self.W[-1] + self.b[-1])[:, 0]

    def _residual(self, x, t, h=0.005):
        u = self.forward(x, t)
        u_t = (self.forward(x, np.minimum(t + h, 1.0))
               - self.forward(x, np.maximum(t - h, 0.0))) / (2 * h)
        u_xx = (self.forward(np.minimum(x + h, 1.0), t)
                - 2 * u
                + self.forward(np.maximum(x - h, -1.0), t)) / h ** 2
        uc = np.clip(u, -2, 2)
        return np.clip(u_t - self.eps ** 2 * u_xx + uc ** 3 - uc, -1e6, 1e6)

    def _loss(self, p, xp, tp, xi, ti, ui, w_ic=20.0):
        self._unpack(p)
        try:
            l_pde = np.mean(self._residual(xp, tp) ** 2)
            l_ic = np.mean((self.forward(xi, ti) - ui) ** 2)
        except Exception:
            return 1e8
        v = l_pde + w_ic * l_ic
        return min(v, 1e8) if np.isfinite(v) else 1e8

    def train(self, xp, tp, xi, ti, ui, epochs=100, verbose=False):
        p = self._pack()
        cc = [0]

        def cb(p):
            cc[0] += 1
            if verbose and cc[0] % 25 == 0:
                print(f"      iter {cc[0]:3d}: loss = "
                      f"{self._loss(p, xp, tp, xi, ti, ui):.6e}")

        res = minimize(lambda p: self._loss(p, xp, tp, xi, ti, ui),
                       p, method="L-BFGS-B", callback=cb,
                       options={"maxiter": epochs, "ftol": 1e-5,
                                "maxfun": epochs * 2})
        self._unpack(res.x)
        return res.fun

    @property
    def n_params(self):
        n = sum(w.size for w in self.W) + sum(b.size for b in self.b)
        return n + sum(len(la) * self.order for la in self.acts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameter sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    eps = 0.01
    T = 0.2

    # ── Sweep settings (reduced budget for speed) ─────────────────────────
    N_pde_sweep = 200
    N_ic_sweep = 50
    epochs_sweep = 80
    n_hidden = 16

    # ── Full-budget settings for final winner ─────────────────────────────
    N_pde_full = 400
    N_ic_full = 100
    epochs_full = 200

    # ── Parameter grid ────────────────────────────────────────────────────
    orders = [3, 4, 5, 6, 8]
    alphas = [-0.5, 0.0, 0.5, 1.0]
    betas  = [-0.5, 0.0, 0.5, 1.0]
    n_total = len(orders) * len(alphas) * len(betas)

    print("=" * 76)
    print("  PIKAN-Jacobi Parameter Sweep  —  Allen-Cahn")
    print("=" * 76)
    print(f"  PDE : u_t = ε² u_xx − (u³ − u),  ε = {eps}")
    print(f"  Sweep grid : order ∈ {orders}")
    print(f"               α     ∈ {alphas}")
    print(f"               β     ∈ {betas}")
    print(f"  Total configs: {n_total}")
    print(f"  Sweep budget : N_pde={N_pde_sweep}, N_ic={N_ic_sweep}, "
          f"epochs={epochs_sweep}")
    print(f"  Final budget : N_pde={N_pde_full}, N_ic={N_ic_full}, "
          f"epochs={epochs_full}")
    print("=" * 76, "\n")

    # ── FD reference ──────────────────────────────────────────────────────
    print("Computing FD reference …")
    x_fd, t_fd, u_fd = fd_reference(eps, T)

    # ── shared training data (sweep) ──────────────────────────────────────
    np.random.seed(0)
    xp = np.random.uniform(-1, 1, N_pde_sweep)
    tp = np.random.uniform(0.001, T, N_pde_sweep)
    xi = np.linspace(-1, 1, N_ic_sweep)
    ti = np.zeros(N_ic_sweep)
    ui = u0_func(xi)

    # ── evaluation helpers ────────────────────────────────────────────────
    xs_eval = np.linspace(-1, 1, 100)
    snap_t = np.linspace(0, T, 5)
    fd_snaps = []
    for tv in snap_t:
        idx = np.argmin(np.abs(t_fd - tv))
        fd_snaps.append(np.interp(xs_eval, x_fd, u_fd[idx]))

    def rel_l2(solver):
        preds = []
        for tv in snap_t:
            preds.append(np.clip(solver.forward(xs_eval, np.full_like(xs_eval, tv)), -5, 5))
        num = sum(np.sum((p - r) ** 2) for p, r in zip(preds, fd_snaps))
        den = sum(np.sum(r ** 2) for r in fd_snaps) + 1e-12
        return np.sqrt(num / den)

    # ── sweep ─────────────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    for i, (order, alpha, beta) in enumerate(
            itertools.product(orders, alphas, betas), 1):
        tag = f"order={order}, α={alpha:+.1f}, β={beta:+.1f}"
        print(f"  [{i:2d}/{n_total}] {tag} …", end="", flush=True)

        t0 = time.time()
        solver = PIKANJacobi(eps, n_hidden=n_hidden,
                             order=order, alpha=alpha, beta=beta)
        loss = solver.train(xp, tp, xi, ti, ui, epochs=epochs_sweep)
        l2 = rel_l2(solver)
        wall = time.time() - t0

        results.append({
            "order": order, "alpha": alpha, "beta": beta,
            "loss": loss, "l2": l2, "time": wall,
            "n_params": solver.n_params,
        })
        print(f"  loss={loss:.4e}  L2={l2:.4e}  ({wall:.1f}s)")

    sweep_time = time.time() - t_start

    # ── sort by L2 error ──────────────────────────────────────────────────
    results.sort(key=lambda r: r["l2"])

    print("\n" + "=" * 76)
    print("  TOP 10 CONFIGURATIONS  (sorted by L₂ error vs FD)")
    print("=" * 76)
    print(f"{'#':>3}  {'order':>5}  {'α':>5}  {'β':>5}  "
          f"{'loss':>12}  {'L₂ err':>12}  {'#params':>7}  {'time(s)':>7}")
    print("-" * 76)
    for rank, r in enumerate(results[:10], 1):
        print(f"{rank:3d}  {r['order']:5d}  {r['alpha']:+5.1f}  {r['beta']:+5.1f}  "
              f"{r['loss']:12.4e}  {r['l2']:12.4e}  {r['n_params']:7d}  "
              f"{r['time']:7.1f}")
    print("=" * 76)
    print(f"\nTotal sweep time: {sweep_time:.0f}s\n")

    # ── re-train winner with full budget ──────────────────────────────────
    best = results[0]
    print("=" * 76)
    print(f"  RE-TRAINING BEST CONFIG WITH FULL BUDGET")
    print(f"  order={best['order']}, α={best['alpha']:+.1f}, "
          f"β={best['beta']:+.1f}")
    print("=" * 76)

    np.random.seed(0)
    xp_f = np.random.uniform(-1, 1, N_pde_full)
    tp_f = np.random.uniform(0.001, T, N_pde_full)
    xi_f = np.linspace(-1, 1, N_ic_full)
    ti_f = np.zeros(N_ic_full)
    ui_f = u0_func(xi_f)

    winner = PIKANJacobi(eps, n_hidden=n_hidden,
                         order=best["order"],
                         alpha=best["alpha"],
                         beta=best["beta"])
    t0 = time.time()
    final_loss = winner.train(xp_f, tp_f, xi_f, ti_f, ui_f,
                              epochs=epochs_full, verbose=True)
    final_time = time.time() - t0
    final_l2 = rel_l2(winner)

    print(f"\n  Final loss : {final_loss:.6e}")
    print(f"  Final L₂   : {final_l2:.6e}")
    print(f"  Time       : {final_time:.1f}s")
    print(f"  # Params   : {winner.n_params}")

    # ── plots ─────────────────────────────────────────────────────────────

    # 1) Heatmap: L2 error for each (alpha, beta) at the best order
    best_order = best["order"]
    res_best_order = [r for r in results if r["order"] == best_order]

    l2_grid = np.full((len(alphas), len(betas)), np.nan)
    for r in res_best_order:
        ai = alphas.index(r["alpha"])
        bi = betas.index(r["beta"])
        l2_grid[ai, bi] = r["l2"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    im = ax.imshow(l2_grid, origin="lower", cmap="viridis_r",
                   aspect="auto",
                   extent=[betas[0] - 0.25, betas[-1] + 0.25,
                           alphas[0] - 0.25, alphas[-1] + 0.25])
    plt.colorbar(im, ax=ax, label="Rel. L₂ error")
    ax.set_xlabel("β"); ax.set_ylabel("α")
    ax.set_title(f"L₂ error  (order={best_order})", fontweight="bold")
    ax.set_xticks(betas); ax.set_yticks(alphas)
    # mark best
    ax.plot(best["beta"], best["alpha"], "r*", ms=20,
            label=f"best: α={best['alpha']}, β={best['beta']}")
    ax.legend(fontsize=9)

    # 2) L2 error vs order (at best alpha, beta)
    ax = axes[1]
    res_best_ab = [r for r in results
                   if r["alpha"] == best["alpha"] and r["beta"] == best["beta"]]
    res_best_ab.sort(key=lambda r: r["order"])
    ord_vals = [r["order"] for r in res_best_ab]
    l2_vals = [r["l2"] for r in res_best_ab]
    ax.plot(ord_vals, l2_vals, "o-", lw=2, ms=8, color="tab:blue")
    ax.set_xlabel("Jacobi order"); ax.set_ylabel("Rel. L₂ error")
    ax.set_title(f"L₂ vs order  (α={best['alpha']}, β={best['beta']})",
                 fontweight="bold")
    ax.set_xticks(ord_vals); ax.grid(True, alpha=0.3)

    # 3) Winner solution snapshots vs FD
    ax = axes[2]
    for i, tv in enumerate(snap_t):
        ts = np.full_like(xs_eval, tv)
        u_w = np.clip(winner.forward(xs_eval, ts), -5, 5)
        ax.plot(xs_eval, fd_snaps[i], "k--", lw=1, alpha=0.5)
        ax.plot(xs_eval, u_w, lw=2, label=f"t={tv:.3f}")
    ax.plot([], [], "k--", lw=1, alpha=0.5, label="FD ref")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.set_title(f"Best PIKAN-Jacobi  (order={best['order']}, "
                 f"α={best['alpha']}, β={best['beta']})", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-1.5, 1.5)

    fig.suptitle("PIKAN-Jacobi Parameter Sweep Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("jacobi_sweep_results.png", dpi=150, bbox_inches="tight")
    print("\nSaved → jacobi_sweep_results.png")
    plt.close()

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  OPTIMAL PIKAN-Jacobi CONFIGURATION")
    print("=" * 76)
    print(f"  order = {best['order']}")
    print(f"  α     = {best['alpha']}")
    print(f"  β     = {best['beta']}")
    print(f"  L₂ error (full budget) = {final_l2:.6e}")
    print(f"  Loss     (full budget) = {final_loss:.6e}")
    print("=" * 76, "\n")


if __name__ == "__main__":
    main()
