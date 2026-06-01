import pytest
import numpy as np
from model_solver import MacrophageModel, Params

class TestModelPhysics:
    """Tests conservation of mass when boundary influx and efflux are switched off"""
    
    @pytest.fixture
    def conservation_model(self):
        # Parameters configured specifically for the mass conservation test
        # gamma, sigma_b, and sigma_max are explicitly 0 to isolate the core domain
        p = Params(
            J=10,              
            N=10, 
            kplus=1.0, 
            kminus=1.0, 
            D_min=0.1, 
            D_max=0.1,           
            gamma=0.0, 
            sigma_b=0.0, 
            sigma_max=0.0, 
            sigma_l=0.0,        
            sigma_w=0.0, 
            sigma_H=0.0, 
            alpha=1.0,
            V_tot=1.0, 
            v_max=1.0, 
            theta=1.0
        )
        return MacrophageModel(p)

    def test_mass_conservation(self, conservation_model):
        """Tests that spatial and structural transitions conserve mass (modulo natural decay)"""
        J = conservation_model.p.J
        N = conservation_model.p.N
        
        # Expand the initial conditions array to include fluid columns: (J + 1, N + 4)
        u0 = np.zeros((J + 1, N + 4))
        
        # Place seed macrophages at a central spatial node (j = J // 2) and empty lipid class (n = 0)
        cell_fraction = 0.8
        u0[J // 2, 0] = cell_fraction  # All mass in the first lipid class for simplicity
        
        # Initialize water (w) to 1.0 across the domain so phi = l + w > 0.
        # This prevents division-by-zero errors in the fluid velocity equations.
        water_fraction = 0.8
        u0[:, N + 1] = 1.0 * (1.0 - water_fraction)             # LDL column (n = N + 1)
        u0[:, N + 2] = 1.0 * water_fraction                  # Empty nodes are 100% water
        u0[J // 2, N + 1] = (1.0 - cell_fraction) * (1.0 - water_fraction)  # Adjust LDL at the seeded node to maintain phi > 0
        u0[J // 2, N + 2] = (1.0 - cell_fraction) * water_fraction  # Adjust water at the seeded node to maintain phi > 0
        
        t_span = (0, 2.0)
        t_eval = np.linspace(*t_span, 15)
        
        # Solve the system
        U, sol = conservation_model.solve(t_span, t_eval, y0=u0.ravel())
        
        # Compute diagnostics to extract domain totals
        diagnostics = conservation_model.compute_diagnostics(U)
        total_macrophages = diagnostics["totals"]["macrophage_number"]
        
        # 4. Mathematically eliminate the hardcoded e^(-t) baseline cell death clearance 
        # to verify that zero mass is lost to numerical diffusion or shifting errors.
        adjusted_mass = total_macrophages * np.exp(t_eval)
        
        initial_mass = adjusted_mass[0]
        assert np.allclose(adjusted_mass, initial_mass, rtol=1e-4), (
            f"Mass conservation broken! Max deviation: {np.max(np.abs(adjusted_mass - initial_mass)):.2e}"
        )