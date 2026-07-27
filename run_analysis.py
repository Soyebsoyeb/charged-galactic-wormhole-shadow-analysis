# Complete analysis runner for charged galactic wormhole.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dark_matter import DarkMatterProfile
from src.shape_function import ShapeFunction
from src.metric import ChargedWormholeMetric
from src.energy_conditions import EnergyConditions
from src.visualization import plot_shape_functions, plot_energy_conditions, plot_metric_components

def main():
    print("=" * 60)
    print("Charged Galactic Wormhole Analysis")
    print("Based on Rahaman et al. (2025), arXiv:2503.16111")
    print("=" * 60)
    
    config_path = 'config/params.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rho_s = config['dark_matter']['rho_s']
    r_s = config['dark_matter']['r_s']
    C1 = config['wormhole']['C1']
    Q = config['wormhole']['Q']
    r_min = config['analysis']['r_min']
    r_max = config['analysis']['r_max']
    n_points = config['analysis']['n_points']
    
    print(f"\nParameters:")
    print(f"  ρ_s = {rho_s}, r_s = {r_s}, C₁ = {C1}, Q = {Q}")
    
    dm = DarkMatterProfile(rho_s, r_s)
    sf = ShapeFunction(rho_s, r_s, C1, Q)
    metric = ChargedWormholeMetric(sf)
    ec = EnergyConditions(sf)
    
    r_values = np.linspace(r_min, r_max, n_points)
    
    print("\n1. Shape Function Analysis...")
    plot_shape_functions(sf, r_values, save_path='plots/shape_functions/shape_analysis.png')
    
    throat = sf.find_throat()
    if throat:
        print(f"   Throat: r₀ = {throat:.4f}, b_eff'(r₀) = {sf.b_eff_prime(throat):.4f}")
    
    print("\n2. Energy Conditions Analysis...")
    plot_energy_conditions(ec, r_values, save_path='plots/energy_conditions/energy_conditions.png')
    
    print("\n3. Metric Analysis...")
    plot_metric_components(metric, r_values, save_path='plots/metric_components.png')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs('plots/shape_functions', exist_ok=True)
    os.makedirs('plots/energy_conditions', exist_ok=True)
    os.makedirs('plots/shadows/static', exist_ok=True)
    os.makedirs('plots/shadows/rotating', exist_ok=True)
    os.makedirs('plots/shadows/plasma', exist_ok=True)
    os.makedirs('plots/comparisons/kerr_vs_wormhole', exist_ok=True)
    os.makedirs('plots/comparisons/plasma_comparisons', exist_ok=True)
    os.makedirs('plots/comparisons/parameter_space', exist_ok=True)
    os.makedirs('plots/comparisons/observational', exist_ok=True)
    main()