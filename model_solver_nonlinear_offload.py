from model_solver import MacrophageModel, Params

class NonlinearMacrophageModel(MacrophageModel):
    def __init__(self, p: Params, offloading_type: str = 'quartic'):
        # 1. Initialize the parent class first. 
        super().__init__(p)
        
        # 2. Calculate the base fractional load (n/N)
        fractional_load = self.n_array / p.N
        
        # 3. Overwrite the offloading multiplier (n_growth) based on your choice
        if offloading_type == 'linear':
            self.n_growth = fractional_load
            
        elif offloading_type == 'quadratic':
            self.n_growth = fractional_load**2
            
        elif offloading_type == 'parabolic':
            # 4 * (n/N) * (1 - n/N)
            self.n_growth =  4.0 *fractional_load * (1.0 - fractional_load)
            
        elif offloading_type == 'quartic':
            # 16 * (n/N)^2 * (1 - n/N)^2
            self.n_growth =  16.0 * (fractional_load**2) * ((1.0 - fractional_load)**2)
            
        else:
            raise ValueError(f"Unknown offloading_type: {offloading_type}")