#!/usr/bin/env python3
"""
Critical-layer numerical experiment for the discrete space/structure model.

Uses the existing MacrophageModel in model_solver.py and keeps the distinguished
limit N ~ M by refining M and N together. The script:
  1. solves a sequence of (M,N) grids with fixed N/M ratio;
  2. locates x_c from phi(x_c) = phi_c = 128/135 for the current kinetics;
  3. extracts the structure profile at the grid point nearest x_c;
  4. measures mass near s_c=2/3 and near s=1;
  5. measures RMS widths of the two peaks;
  6. evaluates the separate discrete RHS contributions near (x_c,s_c):
       - structure jump term
       - physical-space diffusion term
       - loss term
       - the full RHS residual
  7. writes a CSV summary and diagnostic plots.

IMPORTANT:
- The solver's `M` is the number of spatial intervals, with M+1 nodes.
- The solver's `N` is the number of structure intervals, with N+1 nodes.
- We keep N/M fixed throughout the refinement study.
- The model solver itself uses zero initial data when y0 is omitted, so this
  reproduces the source-driven setup in the notebook.
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from model_solver import MacrophageModel, Params


# -------------------------
# User controls
# -------------------------
BASE_PARAMS = Params(
    M=100,
    N=100,
    kplus=5,
    kminus=2,
    D_max=0.01,
    D_min=0.001,
    gamma=0.2,
    sigma_b=0.005,
    sigma_max=0.5,
    V_tot=100000,
    v_max=10,
    theta=10,
    debug=False,
    tol=1e-6,
)

T_FINAL = 15.0
N_TIME = 400

# Keep N/M fixed. Baseline is N=M.
M_VALUES = [10, 20, 40, 80, 160]
N_OVER_M = 1.0

# Peak windows. These are deliberately broad so they are not grid sensitive.
SC = 2.0 / 3.0
PHI_C = 128.0 / 135.0
CRITICAL_WINDOW = (0.50, 0.82)
BOUNDARY_WINDOW = (0.85, 1.00)

OUTPUT_DIR = "critical_layer_experiment_results"


# -------------------------
# Model kinetics
# -------------------------
def offloading_func(n_array: np.ndarray, N: int) -> np.ndarray:
    s = n_array / N
    return 16.0 * s**2 * (1.0 - s)**2


def uptake_func(n_array: np.ndarray, N: int) -> np.ndarray:
    s = n_array / N
    return 1.0 - s


# -------------------------
# Helpers
# -------------------------
def build_params(M: int, N: int) -> Params:
    return replace(BASE_PARAMS, M=M, N=N)


def locate_xc(phi: np.ndarray, x: np.ndarray, phi_c: float = PHI_C) -> tuple[float, int]:
    """Interpolate the first crossing of phi(x)=phi_c, assuming monotone crossing."""
    f = phi - phi_c
    crossings = np.where(f[:-1] * f[1:] <= 0)[0]
    if len(crossings) == 0:
        j = int(np.argmin(np.abs(f)))
        return float(x[j]), j

    j = int(crossings[0])
    if f[j] == 0:
        return float(x[j]), j
    if f[j + 1] == 0:
        return float(x[j + 1]), j + 1

    # Linear interpolation for x_c.
    xc = np.interp(phi_c, phi[j:j+2], x[j:j+2])
    return float(xc), j if abs(x[j] - xc) <= abs(x[j + 1] - xc) else j + 1


def profile_at_xc(U_final: np.ndarray, model: MacrophageModel, phi_final: np.ndarray) -> dict:
    M, N = model.p.M, model.p.N
    x = np.arange(M + 1) / M
    s = np.arange(N + 1) / N

    xc, jc = locate_xc(phi_final, x)

    # Interpolate the full structure profile to the exact x_c.
    # jc is still retained as the nearest grid point for diagnostics
    # that must be evaluated on the discrete grid.
    if jc == 0:
        i_left, i_right = 0, 1
    elif jc == len(x) - 1:
        i_left, i_right = len(x) - 2, len(x) - 1
    else:
        # Find the two grid points that bracket x_c.
        i_right = int(np.searchsorted(x, xc))
        i_left = i_right - 1

        # Safety clipping
        i_left = max(0, min(i_left, len(x) - 2))
        i_right = i_left + 1

    x_left = x[i_left]
    x_right = x[i_right]

    theta = (xc - x_left) / (x_right - x_left)

    u = (
        (1.0 - theta) * U_final[i_left, :]
        + theta * U_final[i_right, :]
    )

    # The notebook's discrete u is mass per structure state. Convert to a
    # continuum-density-like quantity only for plotting and peak-height diagnostics.
    m = N * u

    # Piecewise/trapezoidal integral is more appropriate for a density-like m.
    # The cell mass represented by u is used for peak mass diagnostics.
    def mask(a: float, b: float) -> np.ndarray:
        return (s >= a) & (s <= b)

    mc = mask(*CRITICAL_WINDOW)
    m1 = mask(*BOUNDARY_WINDOW)

    mass_c = float(np.sum(u[mc]))
    mass_1 = float(np.sum(u[m1]))

    def weighted_width(mask_: np.ndarray, center: float) -> float:
        weights = np.maximum(u[mask_], 0.0)
        denom = np.sum(weights)
        if denom <= 0:
            return np.nan
        return float(np.sqrt(np.sum(weights * (s[mask_] - center) ** 2) / denom))

    width_c = weighted_width(mc, SC)
    width_1 = weighted_width(m1, 1.0)

    # Local maxima inside each window.
    ic_local = np.where(mc)[0]
    i1_local = np.where(m1)[0]
    peak_c_idx = int(ic_local[np.argmax(m[ic_local])]) if len(ic_local) else jc
    peak_1_idx = int(i1_local[np.argmax(m[i1_local])]) if len(i1_local) else N

    # No-voids residual at x_c.
    v = model.v
    phi_xc = (

    (1.0 - theta) * phi_final[i_left]

    + theta * phi_final[i_right]

        )

    novoid_residual = float(phi_xc + np.sum(v * u) - 1.0)
    return {
        "xc": xc,
        "jc": jc,
        "phi_xc": float(phi_xc),
        "mass_critical_window": mass_c,
        "mass_boundary_window": mass_1,
        "width_sc": width_c,
        "width_s1": width_1,
        "peak_sc_s": float(s[peak_c_idx]),
        "peak_sc_value": float(m[peak_c_idx]),
        "peak_s1_s": float(s[peak_1_idx]),
        "peak_s1_value": float(m[peak_1_idx]),
        "novoid_residual": novoid_residual,
        "x": x,
        "s": s,
        "u": u,
        "m": m,
    }


def rhs_term_breakdown(model: MacrophageModel, u: np.ndarray, xc_index: int, sc_index: int) -> dict:
    """Evaluate the three main RHS contributions at one grid point."""
    p = model.p
    M, N = p.M, p.N

    phi = model.compute_phi(u)

    i = xc_index
    n = sc_index

    # --- Structure jump term ---
    if 1 <= n <= N - 1:
        structure_term = N * p.kplus * phi[i] * (
            model.uptake[n - 1] * u[i, n - 1] - model.uptake[n] * u[i, n]
        ) + N * p.kminus * (
            model.offloading[n + 1] * u[i, n + 1] - model.offloading[n] * u[i, n]
        )
    elif n == 0:
        structure_term = (
            -N * p.kplus * model.uptake[0] * phi[i] * u[i, 0]
            + N * p.kminus * model.offloading[1] * u[i, 1]
        )
    else:
        structure_term = (
            N * p.kplus * model.uptake[N - 1] * phi[i] * u[i, N - 1]
            - N * p.kminus * model.offloading[N] * u[i, N]
        )

    # --- Physical-space diffusion term ---
    Dn = p.D_min + (p.D_max - p.D_min) * (1.0 - n / N)

    # The model has special boundary stencils in x. Handle them explicitly.
    if i == 0:
        diffusion_term = M**2 * Dn * (
            phi[i] * u[i + 1, n] - phi[i + 1] * u[i, n]
        )
    elif i == M:
        diffusion_term = M**2 * Dn * (
            phi[i] * u[i - 1, n] - phi[i - 1] * u[i, n]
        ) - M * p.gamma * Dn * u[i, n]
    else:
        phi_H = (phi[:, None] * (1.0 - np.arange(M + 1)[:, None] / M >= 0)).astype(float)
        # For the interior critical region the capacity mask is normally one.
        # Reconstruct the exact mask used by rhs().
        capacity_val = (p.V_tot / M) * phi[:, None] - model.v[None, :]
        H = (capacity_val >= 0).astype(float)
        phi_H = phi[:, None] * H
        diffusion_term = M**2 * Dn * (
            phi_H[i, n] * (u[i - 1, n] + u[i + 1, n])
            - u[i, n] * (phi_H[i - 1, n] + phi_H[i + 1, n])
        )

    loss_term = -u[i, n]

    rhs_total = float(model.rhs(0.0, u.ravel())[i * (N + 1) + n])

    return {
        "structure_term": float(structure_term),
        "physical_diffusion_term": float(diffusion_term),
        "loss_term": float(loss_term),
        "rhs_total": rhs_total,
        "phi": float(phi[i]),
        "s": float(n / N),
        "x": float(i / M),
        "D_n": float(Dn),
    }


def run_one(M: int, N: int) -> tuple[dict, dict]:
    print(f"\nRunning M={M}, N={N} (N/M={N/M:.3f})")
    p = build_params(M, N)
    model = MacrophageModel(
        p=p,
        offloading_func=offloading_func,
        uptake_func=uptake_func,
    )

    t_span = (0.0, T_FINAL)
    t_eval = np.linspace(t_span[0], t_span[1], N_TIME)

    # Zero initial condition: source-driven problem, matching solve() default.
    U, sol = model.solve(t_span, t_eval)
    if not sol.success:
        raise RuntimeError(f"Solver failed for M={M}, N={N}: {sol.message}")

    U_final = U[-1]
    phi_final = model.compute_phi(U_final)
    profile = profile_at_xc(U_final, model, phi_final)

    jc = profile["jc"]
    sc_idx = int(round(SC * N))
    sc_idx = min(max(sc_idx, 1), N - 1)
    terms = rhs_term_breakdown(model, U_final, jc, sc_idx)

    row = {
        "M": M,
        "N": N,
        "N_over_M": N / M,
        "t_final": T_FINAL,
        **{k: v for k, v in profile.items() if k not in {"x", "s", "u", "m"}},
        **terms,
        "final_time_solver": float(sol.t[-1]),
    }

    return row, {
        "profile": profile,
        "terms": terms,
        "phi": phi_final,
        "U_final": U_final,
        "solver": sol,
    }


def fit_power_law(xvals: np.ndarray, yvals: np.ndarray) -> float:
    mask = np.isfinite(xvals) & np.isfinite(yvals) & (xvals > 0) & (yvals > 0)
    if np.count_nonzero(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(xvals[mask]), np.log(yvals[mask]), 1)
    return float(slope)


def save_csv(rows: list[dict], path: str) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(all_runs: list[tuple[dict, dict]], outdir: str) -> None:
    rows = [r for r, _ in all_runs]
    Ms = np.array([r["M"] for r in rows], dtype=float)
    widths_c = np.array([r["width_sc"] for r in rows], dtype=float)
    widths_1 = np.array([r["width_s1"] for r in rows], dtype=float)
    mass_c = np.array([r["mass_critical_window"] for r in rows], dtype=float)
    mass_1 = np.array([r["mass_boundary_window"] for r in rows], dtype=float)

    # 1. Peak widths vs M.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ms, widths_c, "o-", label=r"width near $s_c=2/3$")
    ax.loglog(Ms, widths_1, "o-", label=r"width near $s=1$")
    # Candidate slopes for visual comparison; no fitted claim.
    anchor = max(widths_c[0], 1e-12)
    xref = np.array([Ms[0], Ms[-1]], dtype=float)
    ax.loglog(xref, anchor * (xref / Ms[0]) ** (-1/3), "--", label=r"$M^{-1/3}$ reference")
    ax.loglog(xref, anchor * (xref / Ms[0]) ** (-1/2), ":", label=r"$M^{-1/2}$ reference")
    ax.set_xlabel("M")
    ax.set_ylabel("RMS structural width")
    ax.set_title("Structural peak widths under N/M refinement")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "peak_widths_loglog.png"), dpi=180)
    plt.close(fig)

    # 2. Peak masses.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(1.0 / Ms, mass_c, "o-", label=r"mass near $s_c$")
    ax.plot(1.0 / Ms, mass_1, "o-", label=r"mass near $s=1$")
    ax.set_xlabel(r"$1/M$")
    ax.set_ylabel("Discrete mass in window")
    ax.set_title("Mass in the two structural regions")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "peak_masses.png"), dpi=180)
    plt.close(fig)

    # 3. Term magnitudes near critical point.
    fig, ax = plt.subplots(figsize=(7, 5))
    st = np.abs(np.array([r["structure_term"] for r in rows]))
    dx = np.abs(np.array([r["physical_diffusion_term"] for r in rows]))
    loss = np.abs(np.array([r["loss_term"] for r in rows]))
    ax.loglog(Ms, st, "o-", label="structure jump")
    ax.loglog(Ms, dx, "o-", label="physical diffusion")
    ax.loglog(Ms, loss, "o-", label="loss")
    ax.set_xlabel("M")
    ax.set_ylabel("Absolute term magnitude at nearest $(x_c,s_c)$")
    ax.set_title("Discrete balance near the critical point")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "term_balance_loglog.png"), dpi=180)
    plt.close(fig)

    # 4. Final profiles at x_c.
    fig, ax = plt.subplots(figsize=(8, 5))
    for row, data in all_runs:
        prof = data["profile"]
        ax.plot(prof["s"], prof["m"], label=f"M=N={row['M']}")
    ax.axvline(SC, color="black", linestyle="--", alpha=0.6, label=r"$s_c=2/3$")
    ax.set_xlabel("s")
    ax.set_ylabel(r"$N u(x_c,s)$")
    ax.set_title(r"Structure profiles near $x_c$")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "profiles_at_xc.png"), dpi=180)
    plt.close(fig)

    # 5. Spatial phi profiles and phi_c.
    fig, ax = plt.subplots(figsize=(8, 5))
    for row, data in all_runs:
        M = row["M"]
        x = np.arange(M + 1) / M
        ax.plot(x, data["phi"], label=f"M=N={M}")
    ax.axhline(PHI_C, color="black", linestyle="--", alpha=0.6, label=r"$\phi_c=128/135$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x)$")
    ax.set_title("Spatial porosity / free-volume profile")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "phi_profiles.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_runs = []
    for M in M_VALUES:
        N = int(round(N_OVER_M * M))
        row, data = run_one(M, N)
        all_runs.append((row, data))

        print(
            f"  xc={row['xc']:.6g}, phi(xc grid)={row['phi_xc']:.6g}, "
            f"mass_sc={row['mass_critical_window']:.6g}, "
            f"mass_s1={row['mass_boundary_window']:.6g}, "
            f"width_sc={row['width_sc']:.6g}, width_s1={row['width_s1']:.6g}"
        )
        print(
            "  critical-point terms: "
            f"|structure|={abs(row['structure_term']):.3e}, "
            f"|x-diff|={abs(row['physical_diffusion_term']):.3e}, "
            f"|loss|={abs(row['loss_term']):.3e}, "
            f"|rhs|={abs(row['rhs_total']):.3e}"
        )

    rows = [r for r, _ in all_runs]
    save_csv(rows, os.path.join(OUTPUT_DIR, "summary.csv"))
    make_plots(all_runs, OUTPUT_DIR)

    Ms = np.array([r["M"] for r in rows], dtype=float)
    width_c = np.array([r["width_sc"] for r in rows], dtype=float)
    width_1 = np.array([r["width_s1"] for r in rows], dtype=float)

    slope_c = fit_power_law(Ms, width_c)
    slope_1 = fit_power_law(Ms, width_1)

    print("\nEstimated power laws from log-log fits:")
    print(f"  width near s_c ~ M^({slope_c:.4f})")
    print(f"  width near s=1 ~ M^({slope_1:.4f})")
    print(f"\nResults written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
