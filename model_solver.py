# This file defines the MacrophageModel class, which encapsulates the entire atherosclerosis model.
# It includes methods for computing the right-hand side of the ODE system, running the BDF solver, 
# and post-processing the results to compute various diagnostic quantities.

import numpy as np
import warnings
from scipy import spatial
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, kron, diags, lil_matrix
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class Params:
    J: int
    N: int
    kplus: float
    kminus: float
    D_min: float
    D_max: float
    gamma: float
    sigma_b: float
    sigma_max: float
    sigma_l: float
    sigma_w: float
    sigma_H: float
    alpha: float
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
        self.j_array = np.arange(p.J + 1)
        self.n_decay = 1.0 - self.n_array / p.N
        self.n_growth = self.n_array / p.N

        # Default to the standard linear behaviour if no custom functions are provided
        if offloading_func is None:
            offloading_func = lambda n, N: n / N
        if uptake_func is None:
            uptake_func = lambda n, N: 1.0 - n / N
            
        # Generate and store the arrays using the input functions
        self.offloading = offloading_func(self.n_array, self.p.N)
        self.uptake = uptake_func(self.n_array, self.p.N)
        
        # Pre-compute the Jacobian sparsity matrix 
        self.jac_sparsity = self._build_sparsity_matrix()

    def _unpack_state(self, U: np.ndarray):
        """
        Extracts the individual variable arrays from the combined state matrix.
        """
        N = self.p.N
        m = U[..., :N+1]
        l = U[..., N+1]
        w = U[..., N+2]
        H = U[..., N+3]
        phi = l + w
        
        return m, l, w, H, phi

    def compute_phi(self, m: np.ndarray) -> np.ndarray:
        """
        No-voids constraint: phi_j = 1 - sum_n v_n m[j,n]
        """
        phi = 1.0 - np.sum(m * self.v, axis=-1)
        
        min_phi = np.min(phi)
        if min_phi < -self.p.tol:
            warnings.warn(f"Severe physical violation: phi dropped to {min_phi:.2e}. "
                          f"Grid is over-saturated or solver step is too large.")
        
        return np.clip(phi, 0.0, 1.0)

    def compute_VL(self, m: np.ndarray):
        """
        Computes: theta/J * v_max/N * sum_j sum_n (1 - j/J) * v_n * m[j,n]
        """
        j_term = 1 - (self.j_array[:, None] / self.p.J)
        if m.ndim == 3:
            return (self.p.theta / self.p.J) * self.p.delta_v * np.sum(j_term * m * self.n_array, axis=(1, 2))
        else:
            return (self.p.theta / self.p.J) * self.p.delta_v * np.sum(j_term * m * self.n_array)

    def compute_internalised_lipid_volume(self, m: np.ndarray) -> np.ndarray:
        """
        Computes: theta/J * v_max/N * sum_j sum_n v_n * m[j,n]
        """
        if m.ndim == 3:
            return (self.p.theta / self.p.J) * self.p.delta_v * np.sum( m * self.n_array, axis=(1, 2))
        else:
            return (self.p.theta / self.p.J) * self.p.delta_v * np.sum( m * self.n_array)

    def compute_mean_lipid_load_spatial(self, m: np.ndarray) -> np.ndarray:
        """
        Computes:  sum_n(n * m_{j,n}) / sum_n(m_{j,n})
        m shape: (Time, J+1, N+1)
        """
        total_fraction = np.sum(m * self.n_array, axis=2) 
        total_concentration = np.sum(m, axis=2)
        safe_denominator = np.where(total_concentration < 1e-12, 1.0, total_concentration)
        mean_load = (total_fraction / safe_denominator)
        return np.where(total_concentration < 1e-12, 0.0, mean_load)

    def compute_mean_lipid_load_domain(self, m: np.ndarray) -> np.ndarray:
        """
        Computes the domain-wide mean lipid volume fraction over time.
        """
        total_domain_fraction = np.sum(m * self.n_array, axis=(1, 2)) 
        total_domain_macrophages = np.sum(m, axis=(1, 2))
        safe_denominator = np.where(total_domain_macrophages < 1e-12, 1.0, total_domain_macrophages)
        mean_domain_fraction = total_domain_fraction / safe_denominator
        return np.where(total_domain_macrophages < 1e-12, 0.0, mean_domain_fraction)

    def _build_sparsity_matrix(self):
        """
        Build the Jacobian sparsity matrix for the model.

        This version uses a block-tridiagonal spatial structure.
        Within each spatial block, the internal structure is sparsified:
        - Structural states m_n are tridiagonal (depend only on n-1, n, n+1).
        - Macroscopic variables (l, w, H) depend on all m_n, and vice versa.
        """
        p = self.p

        num_spatial = p.J + 1
        block_size = p.N + 4
        N = p.N

        # 1. Spatial coupling (tridiagonal across J)
        # Spatial node j depends on j-1, j, and j+1
        spatial_diags = [
            np.ones(num_spatial - 1),  # lower diagonal
            np.ones(num_spatial),      # main diagonal
            np.ones(num_spatial - 1),  # upper diagonal
        ]

        spatial_sparse = diags(
            spatial_diags,
            offsets=[-1, 0, 1],
            format="csr",
        )

        # 2. Block sparsity (within each spatial node)
        block_sparse = np.zeros((block_size, block_size), dtype=bool)

        # 2a. Macrophage structural states (m_n): tridiagonal dependence
        # m_n depends on m_{n-1}, m_n, and m_{n+1}
        for n in range(N + 1):
            block_sparse[n, n] = True
            if n > 0:
                block_sparse[n, n - 1] = True
            if n < N:
                block_sparse[n, n + 1] = True

        # 2b. Macrophage dependence on macroscopic variables
        # All m_n states depend on LDL (N+1), water (N+2), and HDL (N+3)
        block_sparse[0:N+1, N+1:N+4] = True

        # 2c. Macroscopic variable dependence on macrophages
        # The rates of LDL, water, and HDL depend on sums over all m_n states
        block_sparse[N+1:N+4, 0:N+1] = True

        # 2d. Macroscopic variables depend on each other (and fluid velocity u)
        block_sparse[N+1:N+4, N+1:N+4] = True

        # 3. Kronecker product to build full Jacobian sparsity
        jac_sparsity = kron(
            spatial_sparse,
            block_sparse,
            format="csr",
        )

        return jac_sparsity

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        J, N, p = self.p.J, self.p.N, self.p
        U = y.reshape((J + 1, N + 4)) # N+1 strctured populations, N+2 = LDL, N+3 = water, N+4 = HDL
        m, l, w, H, phi = self._unpack_state(U)
        
        VLt = self.compute_VL(m)

        # FLUID VELOCITY
        u = np.zeros(J + 1)
        Diff_n = p.D_min + (p.D_max - p.D_min) * self.n_decay
        # Pre-calculate the entire summation term for all spatial nodes at once
        flux_sum = np.zeros(J + 1)
        flux_sum[0] = np.sum(J**2 * Diff_n * (phi[0] * m[1, :] - phi[1] * m[0, :]) * self.v)
        flux_internal = (J**2 * Diff_n[None, :] * (phi[1:J, None] * (m[0:J-1, :] + m[2:J+1, :]) - 
                  m[1:J, :] * (phi[0:J-1, None] + phi[2:J+1, None])))
        flux_sum[1:J] = np.sum(flux_internal * self.v, axis=1)
        flux_sum[J] = np.sum((J**2 * Diff_n * (phi[J] * m[J-1, :] - m[J, :] * phi[J-1]) 
                      - J * p.gamma * Diff_n * m[J, :]) * self.v)

        u[0] = ((p.sigma_l + p.sigma_w)/phi[0] + p.sigma_b + 
                p.sigma_max * VLt / (1.0 + VLt) + 1/(J * phi[0]) * flux_sum[0])
        for i in range(1, J + 1):
            u[i] = u[i-1] * phi[i-1] / phi[i] + 1/(J * phi[i]) * flux_sum[i]

        capacity_val = (p.V_tot / J) * phi[:, None] - self.v[None, :]
        Heav = (capacity_val >= 0).astype(float)
        phi_H = phi[:, None] * Heav

        du = np.zeros_like(U)

        # MACROPHAGE EQUATIONS
        # Structure boundaries
        if J >= 2:
            du[1:J, 0] = (
                - N * p.kplus * self.uptake[0] * l[1:J] * m[1:J, 0]
                + N * p.kminus * self.offloading[1] * H[1:J] * m[1:J, 1]
                + J**2 * p.D_max * (phi[1:J] * (m[0:J-1, 0] + m[2:J+1, 0])
                    - m[1:J, 0] * (phi[0:J-1] + phi[2:J+1])
                )
                - m[1:J, 0]
            )

            du[1:J, N] = (
                N * p.kplus * self.uptake[N-1] * l[1:J] * m[1:J, N-1]
                - N * p.kminus * self.offloading[N] * H[1:J] * m[1:J, N]
                + J**2 * p.D_min * (
                    phi[1:J] * (m[0:J-1, N] + m[2:J+1, N])
                    - m[1:J, N] * (phi[0:J-1] + phi[2:J+1])
                )
                - m[1:J, N]
            )

        if N >= 2:
            n = np.arange(1, N)
            D_n = p.D_min + (p.D_max - p.D_min) * self.n_decay[n]

            # Spatial boundaries
            du[0, 1:N] = (
                N * p.kplus * l[0] * (self.uptake[n - 1] * m[0, 0:N-1] - self.uptake[n] * m[0, 1:N])
                + N * p.kminus * H[0] * (self.offloading[n + 1] * m[0, 2:N+1] - self.offloading[n] * m[0, 1:N])
                + J**2 * D_n * (phi[0] * m[1, 1:N] - phi[1] * m[0, 1:N])
                - m[0, 1:N]
            )

            du[J, 1:N] = (
                N * p.kplus * l[J] * (self.uptake[n - 1] * m[J, 0:N-1] - self.uptake[n] * m[J, 1:N])
                + N * p.kminus * H[J] * (self.offloading[n + 1] * m[J, 2:N+1] - self.offloading[n] * m[J, 1:N])
                + J**2 * D_n * (phi[J] * m[J-1, 1:N] - phi[J-1] * m[J, 1:N] )
                - J * p.gamma * D_n * m[J, 1:N]
                - m[J, 1:N]
            )

            # Interior
            if J >= 2:
                D_n_2d = p.D_min + (p.D_max - p.D_min) * self.n_decay[n][None, :]
                du[1:J, 1:N] = (
                    N * p.kplus * l[1:J, None] * (
                        self.uptake[n - 1][None, :] * m[1:J, 0:N-1]
                        - self.uptake[n][None, :] * m[1:J, 1:N]
                    )
                    + N * p.kminus * H[1:J, None] * (
                        self.offloading[n + 1][None, :] * m[1:J, 2:N+1]
                        - self.offloading[n][None, :] * m[1:J, 1:N]
                    )
                    + J**2 * D_n_2d * (
                        phi[1:J, None] * (m[0:J-1, 1:N] + m[2:J+1, 1:N])
                        - m[1:J, 1:N] * (phi[0:J-1, None] + phi[2:J+1, None])
                    )
                    - m[1:J, 1:N]
                )

        # Corners
        du[0, 0] = (
            - N * p.kplus * self.uptake[0] * l[0] * m[0, 0]
            + N * p.kminus * self.offloading[1] * H[0] * m[0, 1]
            + J * (p.sigma_b + p.sigma_max * VLt / (1.0 + VLt)) * phi[0]
            + J**2 * p.D_max * (phi[0] * m[1, 0] - phi[1] * m[0, 0])
            - m[0, 0]
        )

        du[0, N] = (
            N * p.kplus * self.uptake[N - 1] * l[0] * m[0, N-1]
            - N * p.kminus * self.offloading[N] * H[0] * m[0, N]
            + J**2 * p.D_min * (phi[0] * m[1, N] - phi[1] * m[0, N])
            - m[0, N]
        )

        du[J, 0] = (
            - N * p.kplus * self.uptake[0] * l[J] * m[J, 0]
            + N * p.kminus * self.offloading[1] * H[J] * m[J, 1]
            + J**2 * p.D_max * (phi[J] * m[J-1, 0] - phi[J-1] * m[J, 0])
            - J * p.gamma * p.D_max * m[J, 0]
            - m[J, 0]
        )

        du[J, N] = (
            N * p.kplus  * self.uptake[N - 1] * l[J] * m[J, N-1]
            - N * p.kminus * self.offloading[N] * H[J] * m[J, N]
            + J**2 * p.D_min * (phi[J] * m[J-1, N] - phi[J-1] * m[J, N])
            - J * p.gamma * p.D_min * m[J, N]
            - m[J, N]
        )

        # LDL EQUATIONS
        du[0, N+1] = (J * p.sigma_l - J * u[0] * l[0] - N * p.kplus * p.delta_v * l[0] 
                      * np.sum(self.uptake * m[0, :N+1]))
        du[1:J+1, N+1] = (J * (u[0:J] * l[0:J] - u[1:J+1] * l[1:J+1]) - N * p.kplus * p.delta_v 
                          * l[1:J+1] * np.sum(self.uptake * m[1:J+1, :], axis=1))
        
        # WATER EQUATIONS
        du[0, N+2] = ((J * p.sigma_w - J * u[0] * w[0]) + N * p.kminus * H[0] * p.delta_v 
                      * np.sum(self.offloading * m[0, :]) + np.sum(m[0, :] * self.v))
        du[1:J+1, N+2] = (J * (u[0:J] * w[0:J] - u[1:J+1] * w[1:J+1]) + N * p.kminus * H[1:J+1] * p.delta_v 
                          * np.sum(self.offloading * m[1:J+1, :], axis=1) + np.sum(m[1:J+1, :] * self.v, axis=1))
        
        # HDL EQUATIONS
        du[0, N+3] = (J * p.sigma_H - J * u[0] * H[0] - p.alpha * p.delta_v * N * p.kminus * H[0] 
                      * np.sum(self.offloading * m[0, :]) )
        du[1:J+1, N+3] = (J * (u[0:J] * H[0:J] - u[1:J+1] * H[1:J+1]) - p.alpha * p.delta_v * N * p.kminus 
                          * H[1:J+1] * np.sum(self.offloading * m[1:J+1, :], axis=1) )


        return du.ravel()

    def solve(self, t_span: tuple, t_eval: np.ndarray, y0: np.ndarray = None):
        """
        Runs the BDF solver and returns the resulting 'U' 3D array (time, J+1, N+1) 
        and the full solver output object.
        """
        if y0 is None:
            u0 = np.zeros((self.p.J + 1, self.p.N + 4))
            u0[:,self.p.N+2:self.p.N+4] = 1.0
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
            
        U = sol.y.T.reshape((-1, self.p.J + 1, self.p.N + 4))

        # Run sanity checks on the results
        self.validate_results(U, strict=False)
        return U, sol
    
    def compute_velocity_history(self, U: np.ndarray) -> np.ndarray:
        """
        Reconstructs the fluid velocity 'u' profile over time from the solver output.
        U shape: (Time, J + 1, N + 4)
        Returns: array of shape (Time, J + 1)
        """
        J, N, p = self.p.J, self.p.N, self.p
        num_steps = U.shape[0]
        u_history = np.zeros((num_steps, J + 1))
        
        Diff_n = p.D_min + (p.D_max - p.D_min) * self.n_decay
        
        for t_idx in range(num_steps):
            # Extract the 2D spatial grid for the current time step
            m, l, w, H, phi = self._unpack_state(U[t_idx])
            phi = l + w
            VLt = self.compute_VL(m)
            
            # Compute spatial node fluxes [cite: 63, 64]
            flux_sum = np.zeros(J + 1)
            flux_sum[0] = np.sum(J**2 * Diff_n * (phi[0] * m[1, :] - phi[1] * m[0, :]) * self.v)
            
            if J >= 2:
                flux_internal = (J**2 * Diff_n[None, :] * (phi[1:J, None] * (m[0:J-1, :] + m[2:J+1, :]) - 
                          m[1:J, :] * (phi[0:J-1, None] + phi[2:J+1, None])))
                flux_sum[1:J] = np.sum(flux_internal * self.v, axis=1)
                
            flux_sum[J] = np.sum((J**2 * Diff_n * (phi[J] * m[J-1, :] - m[J, :] * phi[J-1]) 
                          - J * p.gamma * Diff_n * m[J, :]) * self.v)

            # Reconstruct the recursive velocity profile [cite: 60]
            u_history[t_idx, 0] = ((p.sigma_l + p.sigma_w)/phi[0] + p.sigma_b + 
                    p.sigma_max * VLt / (1.0 + VLt) + 1/(J * phi[0]) * flux_sum[0]) # [cite: 57, 60]
            for i in range(1, J + 1):
                u_history[t_idx, i] = u_history[t_idx, i-1] * phi[i-1] / phi[i] + 1/(J * phi[i]) * flux_sum[i] # [cite: 60]
                
        return u_history

    def compute_diagnostics(self, U: np.ndarray) -> dict:
        """
        Takes the raw 3D U matrix from the solver and calculates all 
        spatial, structural, and domain-wide profiles.
        Returns a dictionary containing all diagnostic arrays.
        """
        p = self.p
        v = self.v
        V_site = p.V_tot / p.J

        m, l, w, H, phi = self._unpack_state(U)

        space_and_structure = {
            "cell_density": m,
            "cell_volume_fraction": m * v,
        }

        spatial = {
            "cell_density": np.sum(m, axis=2),
            "volume_fraction": np.sum(m * v, axis=2),
            "mean_lipid_load": self.compute_mean_lipid_load_spatial(m),
            "LDL_volume_fraction": l,
            "water_volume_fraction": w,
            "HDL_capacity": H,
            "phi": phi,
            "fluid_velocity": self.compute_velocity_history(U)
        }
        spatial["macrophage_number"] = V_site * spatial["cell_density"]

        structure = {
            "average_cell_density": np.sum(m, axis=1)/self.p.J,
            "tissue_volume_fraction": np.sum(m * v, axis=1)/self.p.J
        }
        structure["macrophage_number"] = p.V_tot * structure["average_cell_density"]

        totals = {
            "average_cell_density": np.sum(m, axis=(1, 2))/p.J,
            "tissue_volume_fraction": np.sum(m * v, axis=(1, 2))/p.J,
            "mean_lipid_load": self.compute_mean_lipid_load_domain(m),
            "internalised_lipid_volume": self.compute_internalised_lipid_volume(m),
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
        Runs rigorous physical and mathematical sanity checks on the solver output.
        If strict=True, raises a ValueError on physical violations.
        Otherwise, raises a warning.
        """
        if not np.isfinite(U).all():
            raise ValueError("NUMERICAL EXPLOSION: NaNs or Infs detected in the output matrix!")
            
        # Global Non-Negativity Check
        min_u = np.min(U)
        if min_u < -self.p.tol:
            msg = f"NON-NEGATIVITY VIOLATION: Minimum concentration value is {min_u:.2e}."
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)

        # Unpack components using our helper method
        m, l, w, H, phi = self._unpack_state(U)

        # Calculate phi from the structural no-voids cell constraint
        phi_no_voids = self.compute_phi(m)

        # Check for deviations between phi computed from the no voids connstriant and from summing l and w
        constraint_dev = np.max(np.abs(phi - phi_no_voids))
        if constraint_dev > self.p.tol:  
            msg = f"PHASE MISMATCH: Fluid volume (l+w) deviates from no-voids constraint! Max dev: {constraint_dev:.2e}"
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)
        
        # Extracellular Volume Fraction Bounds Check
        min_phi, max_phi = np.min(phi), np.max(phi)
        if min_phi < -self.p.tol or max_phi > 1.0 + self.p.tol:
            msg = f"CAPACITY VIOLATION: Fluid fraction phi out of bounds [0, 1]. Min: {min_phi:.2e}, Max: {max_phi:.2e}"
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)

        # NO-VOIDS CONSERVATION CHECK
        # Total volume fraction: fluid fraction (phi) + macrophage volume fraction must equal 1.0
        macro_vol_fraction = np.sum(m * self.v, axis=-1)
        total_volume = phi + macro_vol_fraction
        
        # Calculate maximum absolute deviation from 1.0 across all space and time
        max_dev = np.max(np.abs(total_volume - 1.0))
        if max_dev > self.p.tol:  
            msg = f"CONSERVATION VIOLATION: No-voids constraint broken! Max deviation from 1.0 is {max_dev:.2e}."
            if strict: raise ValueError(msg)
            else: warnings.warn(msg)
            
        # Velocity Explosion Check
        u_hist = self.compute_velocity_history(U)
        max_u = np.max(np.abs(u_hist))
        if max_u > 1e4:  # Threshold adjusted for non-dimensionalised scales
            msg = f"VELOCITY SPIKE WARNING: Maximum fluid velocity reached an extreme value of {max_u:.2e}. System may be near-singular."
            warnings.warn(msg)