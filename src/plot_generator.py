"""
  wormhole shadow analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def plot_shape_function_comparison(sf, r_values, save_path=None):
    """Plot shape function with comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Shape function b(r)
    axes[0, 0].plot(r_values, sf.b(r_values), 'b-', linewidth=3)
    axes[0, 0].set_xlabel('r', fontsize=14)
    axes[0, 0].set_ylabel('b(r)', fontsize=14)
    axes[0, 0].set_title('Shape Function b(r)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Effective shape function
    axes[0, 1].plot(r_values, sf.b_eff(r_values), 'r-', linewidth=3, label='b_eff(r)')
    axes[0, 1].plot(r_values, r_values, 'k--', linewidth=2, label='r')
    axes[0, 1].set_xlabel('r', fontsize=14)
    axes[0, 1].set_ylabel('b_eff(r)', fontsize=14)
    axes[0, 1].set_title('Effective Shape Function', fontsize=14)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Flaring-out condition
    ratio = sf.b_eff(r_values) / r_values
    axes[1, 0].plot(r_values, ratio, 'g-', linewidth=3)
    axes[1, 0].axhline(y=1, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 0].set_xlabel('r', fontsize=14)
    axes[1, 0].set_ylabel('b_eff(r)/r', fontsize=14)
    axes[1, 0].set_title('Flaring-out Condition: b_eff(r)/r < 1', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Derivative
    axes[1, 1].plot(r_values, sf.b_eff_prime(r_values), 'm-', linewidth=3)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 1].set_xlabel('r', fontsize=14)
    axes[1, 1].set_ylabel("b_eff'(r)", fontsize=14)
    axes[1, 1].set_title("Flaring-out: b_eff'(r) < 1", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Shape Function Analysis', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_energy_conditions_comparison(ec, r_values, save_path=None):
    """Plot energy conditions with proper graphs."""
    rho = np.array([ec.rho(r) for r in r_values])
    pr = np.array([ec.pressure_radial(r) for r in r_values])
    pt = np.array([ec.pressure_tangential(r) for r in r_values])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Energy density
    axes[0, 0].plot(r_values, rho, 'b-', linewidth=3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[0, 0].set_xlabel('r', fontsize=14)
    axes[0, 0].set_ylabel(r'$\rho(r)$', fontsize=14)
    axes[0, 0].set_title('Energy Density', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. NEC Radial
    nec_r = rho + pr
    axes[0, 1].plot(r_values, nec_r, 'r-', linewidth=3, label=r'$\rho + P_r$')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[0, 1].fill_between(r_values, 0, nec_r, where=(nec_r < 0), color='red', alpha=0.3, label='Violation')
    axes[0, 1].set_xlabel('r', fontsize=14)
    axes[0, 1].set_ylabel(r'$\rho + P_r$', fontsize=14)
    axes[0, 1].set_title('NEC (Radial): ρ + P_r ≥ 0', fontsize=14)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. NEC Tangential
    nec_t = rho + pt
    axes[1, 0].plot(r_values, nec_t, 'g-', linewidth=3, label=r'$\rho + P_t$')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 0].fill_between(r_values, 0, nec_t, where=(nec_t < 0), color='red', alpha=0.3, label='Violation')
    axes[1, 0].set_xlabel('r', fontsize=14)
    axes[1, 0].set_ylabel(r'$\rho + P_t$', fontsize=14)
    axes[1, 0].set_title('NEC (Tangential): ρ + P_t ≥ 0', fontsize=14)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. SEC
    sec = rho + pr + 2*pt
    axes[1, 1].plot(r_values, sec, 'm-', linewidth=3, label=r'$\rho + P_r + 2P_t$')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 1].fill_between(r_values, 0, sec, where=(sec < 0), color='red', alpha=0.3, label='Violation')
    axes[1, 1].set_xlabel('r', fontsize=14)
    axes[1, 1].set_ylabel(r'$\rho + P_r + 2P_t$', fontsize=14)
    axes[1, 1].set_title('SEC: ρ + P_r + 2P_t ≥ 0', fontsize=14)
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Energy Conditions Analysis', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_deflection_angle_comparison(impact_params, deflection_data, labels, save_path=None):
    """Plot deflection angle comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (data, label) in enumerate(zip(deflection_data, labels)):
        ax.plot(impact_params, data, '-', color=colors[i % len(colors)], 
                linewidth=3, label=label)
    
    # Schwarzschild limit
    ax.plot(impact_params, 4/impact_params, 'k--', linewidth=2, 
            label='Schwarzschild (4M/b)')
    
    ax.set_xlabel('Impact Parameter b', fontsize=14)
    ax.set_ylabel('Deflection Angle α (radians)', fontsize=14)
    ax.set_title('Deflection Angle Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_shadow_boundary_comparison(boundaries, labels, save_path=None):
    """Plot shadow boundary comparison."""
    fig, ax = plt.subplots(figsize=(9, 9))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, (boundary, label) in enumerate(zip(boundaries, labels)):
        if boundary is not None and len(boundary[0]) > 0:
            ax.plot(boundary[0], boundary[1], '-', color=colors[i % len(colors)], 
                    linewidth=3, label=label)
    
    ax.set_xlabel(r'$\alpha$ (celestial x)', fontsize=14)
    ax.set_ylabel(r'$\beta$ (celestial y)', fontsize=14)
    ax.set_title('Shadow Boundary Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_parameter_space_2d(a_vals, rho_vals, R_values, save_path=None):
    """Plot 2D parameter space contour."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X, Y = np.meshgrid(a_vals, rho_vals)
    contour = ax.contourf(X, Y, R_values, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Shadow Radius R', fontsize=14)
    
    ax.set_xlabel('Spin parameter a', fontsize=14)
    ax.set_ylabel('Plasma density ρ', fontsize=14)
    ax.set_title('Shadow Radius R(a, ρ)', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_parameter_space_3d(a_vals, rho_vals, R_values, save_path=None):
    """Plot 3D parameter space surface."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(a_vals, rho_vals)
    surf = ax.plot_surface(X, Y, R_values, cmap='viridis', 
                           alpha=0.9, edgecolor='none', linewidth=0)
    
    ax.set_xlabel('Spin a', fontsize=14, labelpad=10)
    ax.set_ylabel('Plasma ρ', fontsize=14, labelpad=10)
    ax.set_zlabel('Shadow Radius R', fontsize=14, labelpad=10)
    ax.set_title('3D Parameter Space: R(a, ρ)', fontsize=16)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def plot_kerr_vs_wormhole_shadow(kerr_shadow, wormhole_shadow, save_path=None):
    """Plot side-by-side Kerr vs Wormhole shadows."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(kerr_shadow, cmap='hot', origin='lower')
    axes[0].set_title('Kerr Black Hole (a=0.9)', fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(wormhole_shadow, cmap='hot', origin='lower')
    axes[1].set_title('Charged Wormhole (a=0.9)', fontsize=16)
    axes[1].axis('off')
    
    plt.suptitle('Kerr vs Wormhole Shadow Comparison', fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig
