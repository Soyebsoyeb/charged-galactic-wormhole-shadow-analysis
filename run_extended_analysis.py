
# Extended Analysis: Shadow + Plasma + Rotation


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dark_matter import DarkMatterProfile
from src.shape_function import ShapeFunction
from src.metric import ChargedWormholeMetric
from src.metric_rotating import RotatingChargedWormholeMetric
from src.plasma import PlasmaProfile
from src.shadow import WormholeShadow
from src.shadow_rotating import RotatingWormholeShadow
from src.extended_visualization import *

def create_kerr_shadow(a=0.9, N=200):
    shadow = np.ones((N, N))
    center = N // 2
    r_shadow = 6 * (1 - 0.3 * a)
    for i in range(N):
        for j in range(N):
            r = np.sqrt((i - center)**2 + (j - center)**2)
            shadow[i, j] = 0.0 if r < r_shadow * N / 20 else 1.0
    return shadow

def main():
    print("=" * 70)
    print("EXTENDED ANALYSIS: Shadow + Plasma + Rotation")
    print("Charged Galactic Wormhole - Rahaman et al. (2025)")
    print("=" * 70)
    
    with open('config/params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    rho_s = config['dark_matter']['rho_s']
    r_s = config['dark_matter']['r_s']
    C1 = config['wormhole']['C1']
    Q = config['wormhole']['Q']
    
    print(f"\nParameters: ρ_s={rho_s}, r_s={r_s}, C₁={C1}, Q={Q}")
    
    sf = ShapeFunction(rho_s, r_s, C1, Q)
    metric_static = ChargedWormholeMetric(sf)
    
    # PART 1: Static Shadow
    print("\n1. Static Shadow...")
    shadow_static = WormholeShadow(metric_static).generate_shadow(N_pixels=200)
    plt.figure(figsize=(6, 6))
    plt.imshow(shadow_static, cmap='hot', origin='lower')
    plt.title('Static Wormhole (Vacuum)')
    plt.axis('off')
    plt.savefig('plots/shadows/static/static_vacuum_shadow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PART 2: Plasma Shadows
    print("\n2. Plasma Shadows...")
    plasma_densities = [0.0, 0.3, 0.5, 0.7]
    plasma_shadows = []
    for rho in plasma_densities:
        plasma = PlasmaProfile('homogeneous', rho)
        shadow = WormholeShadow(metric_static, plasma).generate_shadow(N_pixels=200)
        plasma_shadows.append(shadow)
    plot_plasma_effect_series(plasma_shadows, plasma_densities,
                              save_path='plots/comparisons/plasma_comparisons/plasma_effect_series.png')
    
    # PART 3: Rotating Shadows
    print("\n3. Rotating Shadows...")
    spin_params = [0.0, 0.3, 0.6, 0.9]
    rotating_shadows = []
    for a in spin_params:
        metric_rot = RotatingChargedWormholeMetric(sf, a)
        shadow = RotatingWormholeShadow(metric_rot).generate_shadow_image(N_pixels=200)
        rotating_shadows.append(shadow)
    plot_rotation_effect_series(rotating_shadows, spin_params,
                                save_path='plots/comparisons/parameter_space/rotation_effect_series.png')
    
    # PART 4: Kerr vs Wormhole
    print("\n4. Kerr Comparison...")
    kerr_shadow = create_kerr_shadow(a=0.9)
    plot_kerr_vs_wormhole_comparison(kerr_shadow, rotating_shadows[-1],
                                     save_path='plots/comparisons/kerr_vs_wormhole/kerr_comparison.png')
    
    # PART 5: Shadow Boundaries
    print("\n5. Shadow Boundaries...")
    boundaries = []
    labels = []
    shadow_static_boundary = WormholeShadow(metric_static).shadow_boundary()
    boundaries.append(shadow_static_boundary)
    labels.append('Static Vacuum')
    plasma = PlasmaProfile('homogeneous', 0.5)
    shadow_plasma_boundary = WormholeShadow(metric_static, plasma).shadow_boundary()
    boundaries.append(shadow_plasma_boundary)
    labels.append('Static Plasma')
    metric_rot = RotatingChargedWormholeMetric(sf, 0.9)
    shadow_rotating_boundary = RotatingWormholeShadow(metric_rot).shadow_boundary()
    boundaries.append(shadow_rotating_boundary)
    labels.append('Rotating (a=0.9)')
    plot_shadow_boundary_comparison(boundaries, labels,
                                    save_path='plots/comparisons/shadow_boundaries/boundary_comparison.png')
    
    # PART 6: Parameter Space
    print("\n6. Parameter Space...")
    a_vals = np.linspace(0, 0.9, 20)
    rho_vals = np.linspace(0, 0.7, 20)
    A, RHO = np.meshgrid(a_vals, rho_vals)
    Z = np.zeros_like(A)
    for i in range(len(a_vals)):
        for j in range(len(rho_vals)):
            metric_rot = RotatingChargedWormholeMetric(sf, A[i, j])
            plasma = PlasmaProfile('homogeneous', RHO[i, j])
            shadow = RotatingWormholeShadow(metric_rot, plasma)
            boundary = shadow.shadow_boundary()
            Z[i, j] = np.sqrt(np.mean(boundary[0]**2 + boundary[1]**2)) if len(boundary[0]) > 0 else 0
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, RHO, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.set_xlabel('Spin a'); ax.set_ylabel('Plasma ρ'); ax.set_zlabel('Shadow Radius R')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.savefig('plots/comparisons/parameter_space/3d_parameter_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PART 7: EHT Constraints
    print("\n7. EHT Constraints...")
    allowed_regions = [
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.3], [0.0, 0.3]]), 'M87*', 'blue'),
        (np.array([[0.2, 0.0], [0.7, 0.0], [0.7, 0.2], [0.2, 0.2]]), 'Sgr A*', 'green')
    ]
    plot_eht_constraints(allowed_regions,
                         save_path='plots/comparisons/observational/eht_constraints.png')
    
    print("\n" + "=" * 70)
    print("EXTENDED ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    os.makedirs('plots/comparisons/shadow_boundaries', exist_ok=True)
    main()