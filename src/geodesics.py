"""
Null geodesics for charged galactic wormhole.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .metric import ChargedWormholeMetric

class NullGeodesic:
    def __init__(self, metric):
        self.metric = metric
        self.Q = metric.Q
    
    def hamiltonian(self, r, theta, pr, ptheta, pphi, pt):
        g_inv = self.metric.inverse_metric(r, theta)
        return 0.5 * (g_inv['g_tt_inv'] * pt**2 +
                      g_inv['g_rr_inv'] * pr**2 +
                      g_inv['g_thth_inv'] * ptheta**2 +
                      g_inv['g_phiphi_inv'] * pphi**2)
    
    def geodesic_equations(self, s, y):
        r, theta, phi, t, pr, ptheta, pphi, pt = y
        g_inv = self.metric.inverse_metric(r, theta)
        return [g_inv['g_rr_inv'] * pr,
                g_inv['g_thth_inv'] * ptheta,
                g_inv['g_phiphi_inv'] * pphi,
                g_inv['g_tt_inv'] * pt,
                0.0, 0.0, 0.0, 0.0]
    
    def integrate(self, y0, s_span=(0, 100), s_eval=None):
        return solve_ivp(self.geodesic_equations, s_span, y0,
                         method='RK45', t_eval=s_eval, rtol=1e-8, atol=1e-10)