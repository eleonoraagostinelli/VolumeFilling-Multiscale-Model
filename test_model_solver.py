import pytest
import numpy as np
from model_solver import MacrophageModel, Params

class TestModelPhysics:
    """Tests conservation of mass when death, influx and efflux are switched off"""
    
    @pytest.fixture
    def conservation_model(self):
        # Parameters configured specifically for the mass conservation test
        # gamma, beta, sigma_b, sigma_max are explicitly 0
        p = Params(
            M=10, N=10, kplus=1.0, kminus=1.0, 
            beta=0.0, gamma=0.0, sigma_b=0.0, sigma_max=0.0, 
            D_min=0.1, V_tot=1.0, v_max=1.0, theta=1.0
        )
        return MacrophageModel(p)

    def test_mass_conservation(self, conservation_model):
        """Tests that turning off birth/death/emigration conserves total mass"""
        M = conservation_model.p.M
        N = conservation_model.p.N
        
        # Initial conditions: all macrophages in one spatial site (i=M/2) and lipid class (n=0)
        u0 = np.zeros((M + 1, N + 1))
        u0[M // 2, 0] = 1.0 
        
        t_span = (0, 5.0)
        t_eval = np.linspace(*t_span, 20)
        
        # Solve
        U, sol = conservation_model.solve(t_span, t_eval, y0=u0.ravel())
        
        # Compute diagnostics to get the total macrophage count over time
        diagnostics = conservation_model.compute_diagnostics(U)
        total_macrophages = diagnostics["totals"]["macrophage_number"]
        
        # Assert mass remains constant over all evaluated time steps
        initial_mass = total_macrophages[0]
        assert np.allclose(total_macrophages, initial_mass, rtol=1e-5), "Total mass was not conserved!"