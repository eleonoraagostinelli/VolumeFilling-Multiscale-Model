# This file defines the MacrophageModel class, which encapsulates the entire atherosclerosis model.
# It includes methods for computing the right-hand side of the ODE system, running the BDF solver, 
# and post-processing the results to compute various diagnostic quantities.

import numpy as np
import warnings
from scipy.integrate import solve_ivp
from scipy.sparse import kron, diags
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Params:
    M: int
    N: int
    kplus: float
    kminus: float
    beta: float
    D_min: float
    gamma: float
    sigma_b: float
    sigma_max: float
    V_tot: float    
    v_max: float
    delta_v: float = field(init=False) 
    theta: float
    debug: bool = False
    tol: float = 1e-6

    def __post_init__(self):
        self.delta_v = self.v_max / self.N

class MacrophageModel:
    def __init__(self, p: Params, 
                 offloading_func: Callable[[np.ndarray, int], np.ndarray] = None, 
                 uptake_func: Callable[[np.ndarray, int], np.ndarray] = None):
        self.p = p
        # Cache static arrays here to save time during ODE integration
        self.v = 1.0 + np.arange(p.N + 1) * p.delta_v
        self.n_array = np.arange(p.N + 1)
        self.i_array = np.arange(p.M + 1)
        self.n_decay = 1.0 - self.n_array / p.N
        self.n_growth = self.n_array / p.N

        # Default to the standard linear behaviour if no custom functions are provided
        if offloading_func is None:
            offloading_func = lambda n, N: 1.0 - n / N
        if uptake_func is None:
            uptake_func = lambda n, N: n / N
            
        # Generate and store the arrays using the input functions
        self.offloading = offloading_func(self.n_array, self.p.N)
        self.uptake = uptake_func(self.n_array, self.p.N)
        
        # Pre-compute the Jacobian sparsity matrix 
        self.jac_sparsity = self._build_sparsity_matrix()

    def compute_phi(self, u: np.ndarray) -> np.ndarray:
        """
        No-voids constraint: phi_i = 1 - sum_n v_n u[i,n]
        """
        phi = 1.0 - np.sum(u * self.v, axis=-1)
        
        min_phi = np.min(phi)
        if min_phi < -self.p.tol:
            warnings.warn(f"Severe physical violation: phi dropped to {min_phi:.2e}. "
                          f"Grid is over-saturated or solver step is too large.")
        
        return np.clip(phi, 0.0, 1.0)

    def compute_VL(self, u: np.ndarray):
        i_term = 1 - (self.i_array[:, None] / self.p.M)
        if u.ndim == 3:
            return (self.p.theta / self.p.M) * self.p.delta_v * np.sum(i_term * u * self.n_array, axis=(1, 2))
        else:
            return (self.p.theta / self.p.M) * self.p.delta_v * np.sum(i_term * u * self.n_array)

    def compute_mean_lipid_load_spatial(self, U: np.ndarray) -> np.ndarray:
        """
        Computes:  sum_n(n * u_{i,n}) / sum_n(u_{i,n})
        U shape: (Time, M+1, N+1)
        """
        total_fraction = np.sum(U * self.n_array, axis=2) 
        total_concentration = np.sum(U, axis=2)
        safe_denominator = np.where(total_concentration < 1e-12, 1.0, total_concentration)
        mean_load = (total_fraction / safe_denominator)
        return np.where(total_concentration < 1e-12, 0.0, mean_load)

    def compute_mean_lipid_load_domain(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the domain-wide mean lipid volume fraction over time.
        """
        total_domain_fraction = np.sum(U * self.n_array, axis=(1, 2)) 
        total_domain_macrophages = np.sum(U, axis=(1, 2))
        safe_denominator = np.where(total_domain_macrophages < 1e-12, 1.0, total_domain_macrophages)
        mean_domain_fraction = total_domain_fraction / safe_denominator
        return np.where(total_domain_macrophages < 1e-12, 0.0, mean_domain_fraction)

    def _build_sparsity_matrix(self):
        """
        Generates the Jacobian sparsity matrix for the atherosclerosis model.
        """
        p = self.p
        spatial_diags = [np.ones(p.M), np.ones(p.M + 1), np.ones(p.M)]
        spatial_sparse = diags(spatial_diags, offsets=[-1, 0, 1], format='csr')
        lipid_dense = np.ones((p.N + 1, p.N + 1))
        jac_sparsity = kron(spatial_sparse, lipid_dense, format='lil')
        jac_sparsity[0, :] = 1
        return jac_sparsity.tocsr()

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        M, N, p = self.p.M, self.p.N, self.p
        u = y.reshape((M + 1, N + 1))

        phi = self.compute_phi(u)
        VLt = self.compute_VL(u)

        capacity_val = (p.V_tot / M) * phi[:, None] - self.v[None, :]
        H = (capacity_val >= 0).astype(float)
        phi_H = phi[:, None] * H

        du = np.zeros_like(u)

        if M >= 2:
            du[1:M, 0] = (
                - N * p.kplus * self.uptake[0] * phi[1:M] * u[1:M, 0]
                + p.kminus * self.offloading[1] * u[1:M, 1]
                + M**2 * (phi_H[1:M, 0] * (u[0:M-1, 0] + u[2:M+1, 0])
                    - u[1:M, 0] * (phi_H[0:M-1, 0] + phi_H[2:M+1, 0])
                )
                - p.beta * u[1:M, 0]
            )

            du[1:M, N] = (
                N * p.kplus * self.uptake[N-1] * phi[1:M] * u[1:M, N-1]
                - N * p.kminus * self.offloading[N] * u[1:M, N]
                + M**2 * p.D_min * (
                    phi_H[1:M, N] * (u[0:M-1, N] + u[2:M+1, N])
                    - u[1:M, N] * (phi_H[0:M-1, N] + phi_H[2:M+1, N])
                )
                - p.beta * u[1:M, N]
            )

        if N >= 2:
            n = np.arange(1, N)
            D_n = p.D_min + (1.0 - p.D_min) * self.n_decay[n]

            du[0, 1:N] = (
                N * p.kplus * phi[0] * (self.uptake[n - 1] * u[0, 0:N-1] - self.uptake[n] * u[0, 1:N])
                + N * p.kminus * (self.offloading[n + 1] * u[0, 2:N+1] - self.offloading[n] * u[0, 1:N])
                + M**2 * D_n * (phi_H[0, 1:N] * u[1, 1:N] - phi_H[1, 1:N] * u[0, 1:N])
                - p.beta * u[0, 1:N]
            )

            du[M, 1:N] = (
                N * p.kplus * phi[M] * (self.uptake[n - 1] * u[M, 0:N-1] - self.uptake[n] * u[M, 1:N])
                + N * p.kminus * (self.offloading[n + 1] * u[M, 2:N+1] - self.offloading[n] * u[M, 1:N])
                + M**2 * D_n * (phi_H[M, 1:N] * u[M-1, 1:N] - phi_H[M-1, 1:N] * u[M, 1:N] )
                - M * p.gamma * D_n * u[M, 1:N]
                - p.beta * u[M, 1:N]
            )

            if M >= 2:
                D_n_2d = p.D_min + (1.0 - p.D_min) * self.n_decay[n][None, :]
                du[1:M, 1:N] = (
                    N * p.kplus * phi[1:M, None] * (
                        self.uptake[n - 1][None, :] * u[1:M, 0:N-1]
                        - self.uptake[n][None, :] * u[1:M, 1:N]
                    )
                    + N * p.kminus * (
                        self.offloading[n + 1][None, :] * u[1:M, 2:N+1]
                        - self.offloading[n][None, :] * u[1:M, 1:N]
                    )
                    + M**2 * D_n_2d * (
                        phi_H[1:M, 1:N] * (u[0:M-1, 1:N] + u[2:M+1, 1:N])
                        - u[1:M, 1:N] * (phi_H[0:M-1, 1:N] + phi_H[2:M+1, 1:N])
                    )
                    - p.beta * u[1:M, 1:N]
                )

        du[0, 0] = (
            - N * p.kplus * self.uptake[0] * phi[0] * u[0, 0]
            + p.kminus * self.offloading[0] * u[0, 1]
            + M * (p.sigma_b + p.sigma_max * VLt / (1.0 + VLt)) * phi_H[0, 0]
            + M**2 * (phi_H[0, 0] * u[1, 0] - phi_H[1, 0] * u[0, 0])
            - p.beta * u[0, 0]
        )

        du[0, N] = (
            p.kplus * self.uptake[N] * phi[0] * u[0, N-1]
            - N * p.kminus * self.offloading[N] * u[0, N]
            + M**2 * p.D_min * (phi_H[0, N] * u[1, N] - phi_H[1, N] * u[0, N])
            - p.beta * u[0, N]
        )

        du[M, 0] = (
            - N * p.kplus * self.uptake[0] * phi[M] * u[M, 0]
            + p.kminus * self.offloading[0] * u[M, 1]
            + M**2 * (phi_H[M, 0] * u[M-1, 0] - phi_H[M-1, 0] * u[M, 0])
            - M * p.gamma * u[M, 0]
            - p.beta * u[M, 0]
        )

        du[M, N] = (
            p.kplus  * self.uptake[N] * phi[M] * u[M, N-1]
            - N * p.kminus * self.offloading[N] * u[M, N]
            + M**2 * p.D_min * (phi_H[M, N] * u[M-1, N] - phi_H[M-1, N] * u[M, N])
            - M * p.gamma * p.D_min * u[M, N]
            - p.beta * u[M, N]
        )

        return du.ravel()

    def solve(self, t_span: tuple, t_eval: np.ndarray, y0: np.ndarray = None):
        """
        Runs the BDF solver and returns the resulting 'U' 3D array (time, M+1, N+1) 
        and the full solver output object.
        """
        if y0 is None:
            u0 = np.zeros((self.p.M + 1, self.p.N + 1))
            y0 = u0.ravel()

        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="BDF",
            jac_sparsity=self.jac_sparsity,
            rtol=self.p.tol,
            atol=1e-9,
        )
        
        if not sol.success:
            warnings.warn(f"Solver failed: {sol.message}")
            
        U = sol.y.T.reshape((-1, self.p.M + 1, self.p.N + 1))

        # Run sanity checks on the results
        self.validate_results(U, strict=False)
        return U, sol

    def compute_diagnostics(self, U: np.ndarray) -> dict:
        """
        Takes the raw 3D U matrix from the solver and calculates all 
        spatial, structural, and domain-wide profiles.
        Returns a dictionary containing all diagnostic arrays.
        """
        p = self.p
        v = self.v
        V_site = p.V_tot / p.M

        space_and_structure = {
            "volume_fraction": U * v,
        }

        spatial = {
            "cell_density": np.sum(U, axis=2),
            "volume_fraction": np.sum(U * v, axis=2),
            "mean_lipid_load": self.compute_mean_lipid_load_spatial(U),
            "phi": self.compute_phi(U)
        }
        spatial["macrophage_number"] = V_site * spatial["cell_density"]

        structure = {
            "average_cell_density": np.sum(U, axis=1)/self.p.M,
            "tissue_volume_fraction": np.sum(U * v, axis=1)/self.p.M
        }
        structure["macrophage_number"] = p.V_tot * structure["average_cell_density"]

        totals = {
            "average_cell_density": np.sum(U, axis=(1, 2))/p.M,
            "tissue_volume_fraction": np.sum(U * v, axis=(1, 2))/p.M,
            "mean_lipid_load": self.compute_mean_lipid_load_domain(U),
            "internalised_lipid_volume": self.compute_VL(U),
        }
        totals["macrophage_number"] = p.V_tot * totals["average_cell_density"]
        totals["cell_volume"] = p.V_tot * totals["tissue_volume_fraction"]

        return {
            "spatial": spatial,
            "structure": structure,
            "totals": totals,
            "space_and_structure": space_and_structure
        }

    def validate_results(self, U: np.ndarray, strict: bool = False):
        """
        Runs physical sanity checks on the solver output.
        If strict=True, raises a ValueError on physical violations.
        Otherwise, raises a warning.
        """
        # print("--- Running Post-Simulation Sanity Checks ---")
        
        if not np.isfinite(U).all():
            raise ValueError("NUMERICAL EXPLOSION: NaNs or Infs detected in the output matrix!")
            
        min_u = np.min(U)
        if min_u < -self.p.tol:
            msg = f"NON-NEGATIVITY VIOLATION: Minimum concentration is {min_u:.2e}."
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)

        phi = self.compute_phi(U)
        min_phi, max_phi = np.min(phi), np.max(phi)
        
        if min_phi < -self.p.tol:
            msg = f"CAPACITY VIOLATION: Minimum phi dropped to {min_phi:.2e}."
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)
            
        if max_phi > 1.0 + self.p.tol:
            msg = f"CAPACITY VIOLATION: Maximum phi exceeded 1.0 ({max_phi:.2e})."
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)
            
        # print("[\u2713] All physical constraints respected!")
        # print("---------------------------------------------")