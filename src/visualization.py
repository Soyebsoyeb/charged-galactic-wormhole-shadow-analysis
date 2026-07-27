"""
Visualization utilities for charged galactic wormhole analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.0,
})

def plot_shape_functions(shape_func, r_values, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(r_values, shape_func.b(r_values), 'b-', linewidth=2)
    axes[0].set_xlabel('r'); axes[0].set_ylabel('b(r)')
    axes[0].set_title('Shape Function b(r)'); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(r_values, shape_func.b_eff(r_values), 'r-', linewidth=2, label='b_eff(r)')
    axes[1].plot(r_values, r_values, 'k--', linewidth=1.5, label='r')
    axes[1].set_xlabel('r'); axes[1].set_ylabel('b_eff(r)')
    axes[1].set_title('Effective Shape Function'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_energy_conditions(ec, r_values, save_path=None):
    rho = [ec.rho(r) for r in r_values]
    pr = [ec.pressure_radial(r) for r in r_values]
    pt = [ec.pressure_tangential(r) for r in r_values]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(r_values, rho, 'b-'); axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('r'); axes[0, 0].set_ylabel(r'$\rho(r)$'); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(r_values, np.array(rho) + np.array(pr), 'r-')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('r'); axes[0, 1].set_ylabel(r'$\rho + P_r$'); axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(r_values, np.array(rho) + np.array(pt), 'g-')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('r'); axes[1, 0].set_ylabel(r'$\rho + P_t$'); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(r_values, np.array(rho) + np.array(pr) + 2*np.array(pt), 'm-')
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel(r'$\rho + P_r + 2P_t$'); axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Energy Conditions', fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metric_components(metric, r_values, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(r_values, metric.g_tt(r_values), 'b-')
    axes[0].set_xlabel('r'); axes[0].set_ylabel('g_tt(r)')
    axes[0].set_title('g_tt = -(1 + Q²/r²)'); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(r_values, metric.g_rr(r_values), 'r-')
    axes[1].set_xlabel('r'); axes[1].set_ylabel('g_rr(r)')
    axes[1].set_title('g_rr = (1 - b(r)/r + Q²/r²)⁻¹'); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()