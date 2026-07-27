"""
Null geodesics for rotating charged wormhole with plasma.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .metric_rotating import RotatingChargedWormholeMetric
from .plasma import PlasmaProfile

class RotatingNullGeodesic:
    def __init__(self, metric, plasma_profile=None):
        self.metric = metric
        self.plasma = plasma_profile
        self.a = metric.a
        self.Q = metric.Q
    
    def hamiltonian(self, r, theta, pr, ptheta, pphi, pt):
        g_inv = self.metric.inverse_metric(r, theta)
        omega_p2 = self.plasma(r, theta) if self.plasma else 0
        return 0.5 * (g_inv['g_tt_inv'] * pt**2 +
                      2 * g_inv['g_tphi_inv'] * pt * pphi +
                      g_inv['g_rr_inv'] * pr**2 +
                      g_inv['g_thth_inv'] * ptheta**2 +
                      g_inv['g_phiphi_inv'] * pphi**2 +
                      omega_p2)
    
    def geodesic_equations(self, s, y):
        r, theta, phi, t, pr, ptheta, pphi, pt = y
        g_inv = self.metric.inverse_metric(r, theta)
        return [g_inv['g_rr_inv'] * pr,
                g_inv['g_thth_inv'] * ptheta,
                g_inv['g_tphi_inv'] * pt + g_inv['g_phiphi_inv'] * pphi,
                g_inv['g_tt_inv'] * pt + g_inv['g_tphi_inv'] * pphi,
                0.0, 0.0, 0.0, 0.0]
    
    def integrate(self, y0, s_span=(0, 100), s_eval=None):
        return solve_ivp(self.geodesic_equations, s_span, y0,
                         method='RK45', t_eval=s_eval, rtol=1e-8, atol=1e-10)
    
    def find_photon_sphere(self, r_min=1.1, r_max=20.0):
        from scipy.optimize import minimize_scalar
        def effective_potential(r):
            g = self.metric.metric_tensor(r, np.pi/2)
            det = g['g_tt'] * g['g_phiphi'] - g['g_tphi']**2
            omega_p2 = self.plasma(r, np.pi/2) if self.plasma else 0
            return -det / (g['g_phiphi'] * (1 - omega_p2))
        result = minimize_scalar(lambda r: -effective_potential(r),
                                 bounds=(r_min, r_max), method='bounded')
        return result.x