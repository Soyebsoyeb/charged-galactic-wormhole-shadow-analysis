"""
Shadow computation for rotating charged wormhole with plasma.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from .metric_rotating import RotatingChargedWormholeMetric
from .plasma import PlasmaProfile

class RotatingWormholeShadow:
    def __init__(self, metric, plasma_profile=None, observer_inclination=17):
        self.metric = metric
        self.plasma = plasma_profile
        self.theta_o = np.radians(observer_inclination)
        self.a = metric.a
        self.Q = metric.Q
    
    def effective_potential(self, r):
        g = self.metric.metric_tensor(r, np.pi/2)
        det = g['g_tt'] * g['g_phiphi'] - g['g_tphi']**2
        omega_p2 = self.plasma(r, np.pi/2) if self.plasma else 0
        return -det / (g['g_phiphi'] * (1 - omega_p2))
    
    def find_photon_sphere(self, r_min=1.1, r_max=20.0):
        result = minimize_scalar(lambda r: -self.effective_potential(r),
                                 bounds=(r_min, r_max), method='bounded')
        return result.x
    
    def celestial_coordinates(self, r_ph, eta, xi):
        theta_o = self.theta_o
        alpha = -eta / np.sin(theta_o)
        beta_sq = (xi - (eta - self.a)**2 + self.a**2 * np.cos(theta_o)**2 
                   - eta**2 * (1/np.tan(theta_o))**2)
        if beta_sq < 0:
            return None, None
        beta = np.sqrt(beta_sq)
        if self.plasma and self.plasma.profile_type == 'homogeneous':
            rho = self.plasma.density_param
            if rho < 1:
                beta = beta / np.sqrt(1 - rho)
                alpha = alpha / np.sqrt(1 - rho)
            else:
                return None, None
        return alpha, beta
    
    def shadow_boundary(self, n_points=100):
        r_ph = self.find_photon_sphere()
        alpha_vals, beta_vals = [], []
        for xi in np.linspace(-5, 5, n_points):
            eta = -self.a + np.sqrt(xi) if xi > 0 else -self.a
            alpha, beta = self.celestial_coordinates(r_ph, eta, xi)
            if alpha is not None and beta is not None:
                alpha_vals.append(alpha)
                beta_vals.append(beta)
        return np.array(alpha_vals), np.array(beta_vals)
    
    def generate_shadow_image(self, N_pixels=200, alpha_max=10, beta_max=10):
        alpha_grid = np.linspace(-alpha_max, alpha_max, N_pixels)
        beta_grid = np.linspace(-beta_max, beta_max, N_pixels)
        shadow = np.ones((N_pixels, N_pixels))
        r_ph = self.find_photon_sphere()
        r_crit = r_ph
        if self.plasma and self.plasma.profile_type == 'homogeneous':
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