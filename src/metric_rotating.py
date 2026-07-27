"""
Rotating charged wormhole metric (Teo-type)
Extends Rahaman et al. (2025) Eq. (3) with rotation.
"""

import numpy as np
from .shape_function import ShapeFunction

class RotatingChargedWormholeMetric:
    def __init__(self, shape_function, a=0.0):
        self.sf = shape_function
        self.Q = shape_function.Q
        self.a = a
    
    def g_tt(self, r):
        return -(1 + self.Q**2 / r**2)
    
    def g_rr(self, r):
        b = self.sf.b(r)
        return 1 / (1 - b/r + self.Q**2 / r**2)
    
    def g_thth(self, r):
        return r**2
    
    def g_phiphi(self, r, theta):
        return r**2 * np.sin(theta)**2
    
    def omega(self, r):
        return 2 * self.a / r**3
    
    def g_tphi(self, r, theta):
        return -self.omega(r) * r**2 * np.sin(theta)**2
    
    def g_phiphi_rot(self, r, theta):
        return self.g_phiphi(r, theta) * (1 + self.omega(r)**2 * r**2 * np.sin(theta)**2)
    
    def metric_tensor(self, r, theta):
        return {
            'g_tt': self.g_tt(r),
            'g_rr': self.g_rr(r),
            'g_thth': self.g_thth(r),
            'g_phiphi': self.g_phiphi_rot(r, theta),
            'g_tphi': self.g_tphi(r, theta)
        }
    
    def inverse_metric(self, r, theta):
        g = self.metric_tensor(r, theta)
        det = g['g_tt'] * g['g_phiphi'] - g['g_tphi']**2
        return {
            'g_tt_inv': g['g_phiphi'] / det,
            'g_rr_inv': 1 / g['g_rr'],
            'g_thth_inv': 1 / g['g_thth'],
            'g_phiphi_inv': g['g_tt'] / det,
            'g_tphi_inv': -g['g_tphi'] / det
        }