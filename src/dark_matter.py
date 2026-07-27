"""
Dark matter profile from Rahaman et al. (2025)
Eq. (9): ρ_w = ρ_s * exp(-r/r_s)
Eq. (10): ρ^(0) = ρ_w
"""

import numpy as np
import yaml

class DarkMatterProfile:
    def __init__(self, rho_s=None, r_s=None, config_path=None):
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                rho_s = config['dark_matter']['rho_s']
                r_s = config['dark_matter']['r_s']
        if rho_s is None or r_s is None:
            raise ValueError("rho_s and r_s must be provided")
        self.rho_s = rho_s
        self.r_s = r_s
    
    def density(self, r):
        """Eq. (9): ρ_w = ρ_s * e^(-r/r_s)"""
        return self.rho_s * np.exp(-r / self.r_s)
    
    def __call__(self, r):
        return self.density(r)