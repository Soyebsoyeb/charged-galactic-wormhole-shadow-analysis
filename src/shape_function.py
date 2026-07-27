"""
Shape function from Rahaman et al. (2025)
Eq. (11): b(r) = -8r_s(r² + 2rr_s + 2r_s²)e^(-r/r_s)πρ_s + C₁
Eq. (12): b_eff(r) = b(r) - Q²/r
"""

import numpy as np
import sympy as sp
import yaml
from .dark_matter import DarkMatterProfile

class ShapeFunction:
    def __init__(self, rho_s=None, r_s=None, C1=None, Q=None, config_path=None):
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                rho_s = config['dark_matter']['rho_s']
                r_s = config['dark_matter']['r_s']
                C1 = config['wormhole']['C1']
                Q = config['wormhole']['Q']
        if any(v is None for v in [rho_s, r_s, C1, Q]):
            raise ValueError("rho_s, r_s, C1, and Q must be provided")
        self.rho_s = rho_s
        self.r_s = r_s
        self.C1 = C1
        self.Q = Q
        self.dm = DarkMatterProfile(rho_s, r_s)
    
    def b(self, r):
        """Eq. (11): b(r) = -8r_s(r² + 2rr_s + 2r_s²)e^(-r/r_s)πρ_s + C₁"""
        term1 = -8 * self.r_s * (r**2 + 2*r*self.r_s + 2*self.r_s**2)
        
        # Handle both numeric and symbolic input
        if hasattr(r, 'free_symbols'):  # SymPy symbolic
            term2 = sp.exp(-r / self.r_s)
        else:  # Numeric
            term2 = np.exp(-r / self.r_s)
            
        term3 = np.pi * self.rho_s
        return term1 * term2 * term3 + self.C1
    
    def b_symbolic(self, r):
        """Symbolic version of Eq. (11) for SymPy."""
        term1 = -8 * self.r_s * (r**2 + 2*r*self.r_s + 2*self.r_s**2)
        term2 = sp.exp(-r / self.r_s)
        term3 = sp.pi * self.rho_s
        return term1 * term2 * term3 + self.C1
    
    def b_eff(self, r):
        """Eq. (12): b_eff(r) = b(r) - Q²/r"""
        return self.b(r) - self.Q**2 / r
    
    def b_eff_symbolic(self, r):
        """Symbolic version of Eq. (12)."""
        return self.b_symbolic(r) - self.Q**2 / r
    
    def b_prime(self, r):
        """Numeric derivative only."""
        if hasattr(r, 'free_symbols'):
            raise ValueError("b_prime() only works with numeric values")
        factor = -8 * np.pi * self.rho_s * self.r_s
        bracket = (2*r + 2*self.r_s) - (r**2 + 2*r*self.r_s + 2*self.r_s**2) / self.r_s
        return factor * bracket * np.exp(-r / self.r_s)
    
    def b_eff_prime(self, r):
        """Numeric derivative only."""
        if hasattr(r, 'free_symbols'):
            raise ValueError("b_eff_prime() only works with numeric values")
        return self.b_prime(r) + self.Q**2 / r**2
    
    def find_throat(self, r_min=0.1, r_max=10.0):
        from scipy.optimize import root_scalar
        def condition(r): return self.b_eff(r) - r
        try:
            result = root_scalar(condition, bracket=[r_min, r_max])
            return result.root
        except:
            return None
    
    def check_flaring_out(self, r):
        return self.b_eff_prime(r) < 1