"""
Energy conditions from Rahaman et al. (2025)
Eq. (4)-(8) and NEC
"""

import numpy as np

class EnergyConditions:
    def __init__(self, shape_function):
        self.sf = shape_function
        self.Q = shape_function.Q
    
    def rho(self, r):
        """Eq. (4): ρ = (b'/r² + Q²/r⁴) / (8π)"""
        return (self.sf.b_prime(r) / r**2 + self.Q**2 / r**4) / (8 * np.pi)
    
    def rho_0(self, r):
        return self.sf.dm.density(r)
    
    def rho_1(self, r):
        """Eq. (8): ρ^(1) = Q²/(8πr⁴)"""
        return self.Q**2 / (8 * np.pi * r**4)
    
    def tau(self, r):
        """Eq. (5): τ(r)"""
        b = self.sf.b(r)
        term1 = b / r**3 - self.Q**2 / r**4
        term2 = 2 * (1 - b/r + self.Q**2 / r**2) * self.Q**2 / (r**2 * (r**2 + self.Q**2))
        return (term1 + term2) / (8 * np.pi)
    
    def P(self, r):
        """Eq. (6): P(r)"""
        b = self.sf.b(r)
        g_rr_inv = 1 - b/r + self.Q**2 / r**2
        term1 = self.Q**2 * (3*r**2 + self.Q**2) / (r**2 * (r**2 + self.Q**2))
        term2 = (self.sf.b_prime(r)*r - b + 2*self.Q**2/r**2) / (2*(r**2 - b*r + self.Q**2))
        term3 = self.Q**2 / (r * (r**2 + self.Q**2)**2)
        return g_rr_inv * (term1 + term2 * term3 + term3**2 - term3) / (8 * np.pi)
    
    def pressure_radial(self, r):
        return -self.tau(r)
    
    def pressure_tangential(self, r):
        return self.P(r)
    
    def check_NEC(self, r):
        rho = self.rho(r)
        pr = self.pressure_radial(r)
        pt = self.pressure_tangential(r)
        return {
            'radial': rho + pr >= 0,
            'tangential': rho + pt >= 0,
            'violated': (rho + pr < 0) or (rho + pt < 0)
        }
    
    def get_all(self, r):
        rho = self.rho(r)
        pr = self.pressure_radial(r)
        pt = self.pressure_tangential(r)
        return {'rho': rho, 'pr': pr, 'pt': pt, 'rho_pr': rho+pr, 'rho_pt': rho+pt}