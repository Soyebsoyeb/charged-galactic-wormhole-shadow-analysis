"""
Plasma profiles for wormhole shadow analysis.
"""

import numpy as np

class PlasmaProfile:
    def __init__(self, profile_type='homogeneous', density_param=0.0):
        self.profile_type = profile_type
        self.density_param = density_param
    
    def omega_p2(self, r, theta=np.pi/2):
        if self.profile_type == 'homogeneous':
            return self.density_param
        elif self.profile_type == 'longitudinal':
            return self.density_param * (1 + 2 * np.sin(theta)**2)
        elif self.profile_type == 'radial':
            return self.density_param / r**(3/2)
        elif self.profile_type == 'spherical':
            return self.density_param / r**2
        else:
            return 0.0
    
    def __call__(self, r, theta=np.pi/2):
        return self.omega_p2(r, theta)