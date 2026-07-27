"""
Extended visualization for shadow + plasma + rotation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def plot_plasma_effect_series(shadow_images, plasma_densities, save_path=None):
    n = len(shadow_images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, (img, rho) in enumerate(zip(shadow_images, plasma_densities)):
        axes[i].imshow(img, cmap='hot', origin='lower')
        axes[i].set_title(f'ρ = {rho}', fontsize=10)
        axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.suptitle('Plasma Effect on Shadows', fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rotation_effect_series(shadow_images, spin_params, save_path=None):
    n = len(shadow_images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, (img, a) in enumerate(zip(shadow_images, spin_params)):
        axes[i].imshow(img, cmap='hot', origin='lower')
        axes[i].set_title(f'a = {a:.2f}', fontsize=10)
        axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.suptitle('Rotation Effect on Shadows', fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_kerr_vs_wormhole_comparison(kerr_shadow, wormhole_shadow, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(kerr_shadow, cmap='hot', origin='lower')
    axes[0].set_title('Kerr Black Hole (a=0.9)', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(wormhole_shadow, cmap='hot', origin='lower')
    axes[1].set_title('Charged Wormhole (a=0.9)', fontsize=12)
    axes[1].axis('off')
    plt.suptitle('Kerr vs Wormhole Shadow Comparison', fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shadow_boundary_comparison(boundaries, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(boundaries)))
    for i, (boundary, label) in enumerate(zip(boundaries, labels)):
        if boundary is not None and len(boundary[0]) > 0:
            ax.plot(boundary[0], boundary[1], '-', color=colors[i], linewidth=2, label=label)
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$\beta$')
    ax.set_title('Shadow Boundary Comparison'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_eht_constraints(allowed_regions, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for region, label, color in allowed_regions:
        ax.fill(region[:, 0], region[:, 1], alpha=0.3, color=color, label=label)
    ax.set_xlabel('Spin parameter a'); ax.set_ylabel('Plasma density ρ')
    ax.set_title('EHT Constraints on Wormhole Parameters')
    ax.legend(); ax.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()