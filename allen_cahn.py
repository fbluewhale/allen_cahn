"""
Allen–Cahn solver (1D) on [-1, 1] using Jacobi orthogonal polynomials
and a pseudo-spectral Galerkin projection (modal coefficients).

PDE:
    u_t = eps^2 * u_xx - (u^3 - u)

We expand:
    u(x,t) ≈ sum_{n=0}^N a_n(t) * P_n^{(α,β)}(x)

Spatial operators are handled spectrally via Jacobi orthogonality.
Time stepping uses a simple IMEX Euler scheme:
    (I - dt*eps^2*L) a^{n+1} = a^n + dt*(a^n - proj(u^3))

Notes:
- This code uses Gauss–Jacobi quadrature for projection.
- The nonlinear term is handled pseudo-spectrally with optional dealiasing
  by using M > N+1 quadrature nodes (default uses a 3/2 rule).
- Boundary conditions: this formulation is “natural” Galerkin and does NOT
  explicitly enforce Dirichlet values at x=±1. If you need strict Dirichlet,
  use a boundary-adapted basis (e.g., Shen-type) or tau/penalty enforcement.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.special import roots_jacobi, jacobi
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


def build_jacobi_basis_and_derivs(x, N, alpha, beta):
    """
    Returns matrices:
      P   : shape (N+1, M)   where P[n,j] = P_n^{(α,β)}(x_j)
      Pxx : shape (N+1, M)   where Pxx[n,j] = d^2/dx^2 P_n^{(α,β)}(x_j)
    using scipy.special.jacobi which provides an orthopoly1d.
    """
    M = x.size
    P = np.zeros((N + 1, M), dtype=float)
    Pxx = np.zeros((N + 1, M), dtype=float)

    for n in range(N + 1):
        poly = jacobi(n, alpha, beta)          # orthopoly1d for P_n^(α,β)
        poly_xx = poly.deriv(2)                # second derivative polynomial
        P[n, :] = poly(x)
        Pxx[n, :] = poly_xx(x)

    return P, Pxx


def modal_norms(P, w):
    """
    Compute modal norms:
      norm_n = ∫ P_n^2 w(x) dx  ≈ sum_j w_j * P_n(x_j)^2
    where w_j are Gauss–Jacobi quadrature weights (already include the Jacobi weight).
    """
    return (P * P * w[None, :]).sum(axis=1)  # shape (N+1,)


def project_to_modal(values, P, w, norms):
    """
    Project nodal function values (at quadrature nodes) to Jacobi modal coefficients:
      a_n = <f, P_n> / <P_n, P_n>
          ≈ sum_j w_j f(x_j) P_n(x_j) / norm_n
    """
    rhs = (P * (w[None, :] * values[None, :])).sum(axis=1)  # shape (N+1,)
    return rhs / norms


def reconstruct_from_modal(a, P):
    """Reconstruct nodal values from modal coefficients: u(x_j) = sum_n a_n P_n(x_j)."""
    return (a[:, None] * P).sum(axis=0)


def build_laplacian_operator(N, alpha, beta, M_dealias=None):
    """
    Build the modal operator L such that:
      (L a) ≈ projection of u_xx where u = sum a_n P_n.
    This gives u_xx coefficients in the same Jacobi basis.

    We compute L by applying it to each basis vector e_k:
      u = P_k  => u_xx = d^2 P_k/dx^2
      then project u_xx back to modal coefficients.
    """
    if M_dealias is None:
        M = N + 1
    else:
        M = int(M_dealias)

    x, w = roots_jacobi(M, alpha, beta)
    P, Pxx = build_jacobi_basis_and_derivs(x, N, alpha, beta)
    norms = modal_norms(P, w)

    L = np.zeros((N + 1, N + 1), dtype=float)
    # For basis vector e_k, u_xx values are exactly Pxx[k, :]
    for k in range(N + 1):
        uxx_vals = Pxx[k, :]
        L[:, k] = project_to_modal(uxx_vals, P, w, norms)

    return L, x, w, P, norms


def solve_allen_cahn_jacobi(
    eps=0.01,
    T=1.0,
    dt=1e-3,
    N=60,
    alpha=0.0,
    beta=0.0,
    dealias_factor=1.5,
    u0_func=None,
    store_every=100
):
    """
    Main solver.

    Parameters
    ----------
    eps : float
        Interface parameter ε.
    T : float
        Final time.
    dt : float
        Time step size.
    N : int
        Polynomial degree (number of modes is N+1).
    alpha, beta : floats
        Jacobi parameters (α,β). Special cases:
          (0,0)  -> Legendre
          (-1/2,-1/2) -> Chebyshev (Gauss–Chebyshev quadrature differs; keep α,β > -1)
    dealias_factor : float
        Use M = ceil(dealias_factor*(N+1)) quadrature points for nonlinear projection.
        1.5 is a common “3/2 rule” dealiasing choice.
    u0_func : callable
        Initial condition u0(x). If None, uses a standard smooth example.
    store_every : int
        Save snapshots every this many steps for plotting.

    Returns
    -------
    xs_plot : ndarray
        A dense x-grid for plotting.
    snapshots : list of (t, u(xs_plot))
        Solution snapshots.
    """
    if u0_func is None:
        # A common smooth test IC
        u0_func = lambda x: (x**2) * np.cos(np.pi * x)

    # Quadrature for projections (use dealiasing points)
    M = int(np.ceil(dealias_factor * (N + 1)))
    xq, wq = roots_jacobi(M, alpha, beta)

    # Build basis at quadrature nodes
    Pq, Pxxq = build_jacobi_basis_and_derivs(xq, N, alpha, beta)
    norms = modal_norms(Pq, wq)

    # Build Laplacian operator L (modal)
    L, _, _, _, _ = build_laplacian_operator(N, alpha, beta, M_dealias=M)

    # Initial condition -> modal coefficients
    u0_vals = u0_func(xq)
    a = project_to_modal(u0_vals, Pq, wq, norms)

    # IMEX system matrix: (I - dt*eps^2*L)
    I = np.eye(N + 1)
    A = I - dt * (eps**2) * L
    lu, piv = lu_factor(A)

    # For plotting, use a dense grid
    xs_plot = np.linspace(-1, 1, 800)
    Pplot, _ = build_jacobi_basis_and_derivs(xs_plot, N, alpha, beta)

    snapshots = []
    steps = int(np.ceil(T / dt))
    t = 0.0

    def eval_on_plot(a_coeffs):
        return reconstruct_from_modal(a_coeffs, Pplot)

    # store initial
    snapshots.append((t, eval_on_plot(a)))

    for n in range(1, steps + 1):
        t = n * dt

        # reconstruct u at quadrature nodes
        u_vals = reconstruct_from_modal(a, Pq)

        # nonlinear term: u^3 projected back to modal
        u3_vals = u_vals**3
        u3_coeffs = project_to_modal(u3_vals, Pq, wq, norms)

        # IMEX Euler update:
        # (I - dt eps^2 L) a^{n+1} = a^n + dt*(a^n - u3_coeffs)
        rhs = a + dt * (a - u3_coeffs)
        a = lu_solve((lu, piv), rhs)

        if n % store_every == 0 or n == steps:
            snapshots.append((t, eval_on_plot(a)))

    return xs_plot, snapshots


if __name__ == "__main__":
    # Example run
    xs, snaps = solve_allen_cahn_jacobi(
        eps=0.02,
        T=0.2,
        dt=5e-4,
        N=80,
        alpha=0.0,
        beta=0.0,        # Legendre (Jacobi α=β=0)
        dealias_factor=1.5,
        store_every=100
    )

    # Plot a few snapshots
    plt.figure()
    for (t, u) in snaps[::max(1, len(snaps)//6)]:
        plt.plot(xs, u, label=f"t={t:.3f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Allen–Cahn via Jacobi (pseudo-spectral Galerkin + IMEX)")
    plt.legend()
    plt.tight_layout()
    plt.show()
