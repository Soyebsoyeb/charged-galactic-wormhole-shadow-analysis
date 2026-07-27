"""
Charged galactic wormhole metric from Rahaman et al. (2025)
Eq. (3): ds² = -(1 + Q²/r²)dt² + (1 - b(r)/r + Q²/r²)⁻¹dr² + r²(dθ² + sin²θ dφ²)
"""

import numpy as np
from .shape_function import ShapeFunction

class ChargedWormholeMetric:
    def __init__(self, shape_function):
        self.sf = shape_function
        self.Q = shape_function.Q
    
    def g_tt(self, r):
        """Eq. (3): g_tt = -(1 + Q²/r²)"""
        return -(1 + self.Q**2 / r**2)
    
    def g_rr(self, r):
        """Eq. (3): g_rr = (1 - b(r)/r + Q²/r²)⁻¹"""
        b = self.sf.b(r)
        return 1 / (1 - b/r + self.Q**2 / r**2)
    
    def g_thth(self, r):
        return r**2
    
    def g_phiphi(self, r, theta):
        return r**2 * np.sin(theta)**2
    
    def metric_tensor(self, r, theta):
        return {
            'g_tt': self.g_tt(r),
            'g_rr': self.g_rr(r),
            'g_thth': self.g_thth(r),
            'g_phiphi': self.g_phiphi(r, theta)
        }
    
    def inverse_metric(self, r, theta):
        g = self.metric_tensor(r, theta)
        return {
            'g_tt_inv': 1 / g['g_tt'],
            'g_rr_inv': 1 / g['g_rr'],
            'g_thth_inv': 1 / g['g_thth'],
            'g_phiphi_inv': 1 / g['g_phiphi']
        }