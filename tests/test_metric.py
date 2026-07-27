"""
Unit tests for charged galactic wormhole metric.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.metric import ChargedWormholeMetric
from src.shape_function import ShapeFunction
from src.dark_matter import DarkMatterProfile

class TestMetric:
    def setup_method(self):
        self.rho_s, self.r_s, self.C1, self.Q = 0.05, 1.0, 0.0, 0.1
        self.sf = ShapeFunction(self.rho_s, self.r_s, self.C1, self.Q)
        self.metric = ChargedWormholeMetric(self.sf)
    
    def test_g_tt(self):
        r = 2.0
        expected = -(1 + self.Q**2 / r**2)
        assert np.isclose(self.metric.g_tt(r), expected)
    
    def test_g_rr(self):
        r = 2.0
        b = self.sf.b(r)
        expected = 1 / (1 - b/r + self.Q**2 / r**2)
        assert np.isclose(self.metric.g_rr(r), expected)
    
    def test_metric_tensor(self):
        r, theta = 2.0, np.pi/4
        g = self.metric.metric_tensor(r, theta)
        assert 'g_tt' in g and 'g_rr' in g and 'g_thth' in g and 'g_phiphi' in g
        assert np.isclose(g['g_thth'], r**2)
        assert np.isclose(g['g_phiphi'], r**2 * np.sin(theta)**2)

if __name__ == "__main__":
    pytest.main([__file__])