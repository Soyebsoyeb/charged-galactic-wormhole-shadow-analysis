"""
Shadow computation for charged wormhole with plasma.
"""

import numpy as np
from tqdm import tqdm
from .metric import ChargedWormholeMetric
from .plasma import PlasmaProfile

class PlasmaShadow:
    def __init__(self, metric, plasma_profile, observer_inclination=17):
        self.metric = metric
        self.plasma = plasma_profile
        self.theta_o = np.radians(observer_inclination)
        self.Q = metric.Q
    
    def effective_potential(self, r):
        g = self.metric.metric_tensor(r, np.pi/2)
        omega_p2 = self.plasma(r, np.pi/2)
        return -g['g_tt'] * (1 - omega_p2) / g['g_rr']
    
    def find_photon_sphere(self, r_min=1.1, r_max=20.0):
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda r: -self.effective_potential(r),
                                 bounds=(r_min, r_max), method='bounded')
        return result.x
    
    def shadow_boundary(self, n_points=100):
        r_ph = self.find_photon_sphere()
        alpha_vals, beta_vals = [], []
        for xi in np.linspace(-5, 5, n_points):
            eta = xi
            alpha = -eta / np.sin(self.theta_o)
            if self.plasma.profile_type == 'homogeneous':
                rho = self.plasma.density_param
                if rho < 1:
                    scale = 1 / np.sqrt(1 - rho)
                    beta_sq = (xi**2 - eta**2) * scale**2
                else:
                    continue
            else:
                beta_sq = xi**2 - eta**2
            if beta_sq > 0:
                beta_vals.append(np.sqrt(beta_sq))
                alpha_vals.append(alpha)
        return np.array(alpha_vals), np.array(beta_vals)
    
    def generate_shadow_image(self, N_pixels=200, alpha_max=10, beta_max=10):
        alpha_grid = np.linspace(-alpha_max, alpha_max, N_pixels)
        beta_grid = np.linspace(-beta_max, beta_max, N_pixels)
        shadow = np.ones((N_pixels, N_pixels))
        r_ph = self.find_photon_sphere()
        r_crit = r_ph
        if self.plasma.profile_type == 'homogeneous':
            rho = self.plasma.density_param
            if rho < 1:
                r_crit = r_ph / np.sqrt(1 - rho)
            else:
                return shadow
        
        for i, alpha in enumerate(tqdm(alpha_grid)):
            for j, beta in enumerate(beta_grid):
                if np.sqrt(alpha**2 + beta**2) < r_crit:
                    shadow[i, j] = 0.0
        return shadow