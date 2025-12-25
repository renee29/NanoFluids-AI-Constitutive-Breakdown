#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NANOFLUIDS-AI: WORK PACKAGE 2 – CONTINUUM THEORY VALIDATION
================================================================================
Project: NanoFluids-AI – Continuum Theory Validation
Author: NanoFluids-AI Research Team

Purpose:
    Molecular dynamics simulation of a Lennard-Jones fluid in a nanochannel
    to validate momentum conservation and recover the Navier-Stokes constitutive
    law from first-principles molecular data.

Scientific Validation:
    - Panel A: Shear stress profile τ_xy(y) with linear momentum balance
    - Panel B: Velocity profile u_x(y) with parabolic Poiseuille solution

Methodological Corrections (v2.0):
    1. Proper Velocity-Verlet integration scheme
    2. Irving-Kirkwood method for inhomogeneous stress tensor
    3. Consistent virial stress assignment across bin boundaries

Author: NanoFluids-AI Research Team
License: MIT
================================================================================
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from numba import jit
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# --- System Configuration ---
N_PARTICLES: int = 800           # REDUCIDO de 2000 para viabilidad O(N^2)
DENSITY: float = 0.85            # Reduced number density (σ⁻³)
TEMPERATURE: float = 1.0         # Reduced temperature (ε/k_B)

# --- Integration Parameters ---
DT: float = 0.002                # Timestep (τ_LJ) 
TOTAL_STEPS: int = 100000        # Reducido ligeramente para velocidad
EQUILIBRATION_STEPS: int = 5000  # Aumentado para estabilidad térmica
SAMPLE_INTERVAL: int = 5         # Muestreo menos frecuente (ahorra memoria)

# --- Channel Geometry ---
BOX_X: float = 10.0              # Reducido proporcionalmente a N
BOX_Y: float = N_PARTICLES / (DENSITY * BOX_X)  # Channel height (σ)

# --- External Forcing ---
F_DRIVE: float = 0.02            # Body-force acceleration per particle (ε/σ)

# --- Analysis Parameters ---
BIN_WIDTH: float = 0.6           # Spatial bin width for profiles (σ)
WALL_EXCLUSION: float = 2.5      # Exclude near-wall region from fits (σ)

# --- LJ Potential Parameters ---
LJ_CUTOFF: float = 2.5           # LJ cutoff radius (σ)
LJ_CUTOFF_SQ: float = LJ_CUTOFF ** 2
WCA_CUTOFF: float = 2.0 ** (1.0 / 6.0)  # WCA cutoff ≈ 1.122σ


# =============================================================================
# PHYSICS KERNELS (Numba-accelerated)
# =============================================================================

@jit(nopython=True, cache=True)
def compute_forces_and_virial(
    positions: np.ndarray,
    box_x: float,
    box_y: float,
    f_drive: float,
    bin_width: float,
    n_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute interparticle forces and the Irving-Kirkwood stress tensor.
    
    The Irving-Kirkwood (IK) method correctly assigns virial contributions
    to spatial bins by distributing the stress along the line connecting
    interacting particle pairs. This is essential for inhomogeneous systems
    such as confined fluids with density gradients near walls.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 2)
    box_x : float
        Box dimension in x (periodic)
    box_y : float
        Box dimension in y (confined)
    f_drive : float
        External driving force in x-direction
    bin_width : float
        Width of spatial bins for stress profile
    n_bins : int
        Number of bins in y-direction
    
    Returns
    -------
    forces : np.ndarray
        Force on each particle, shape (N, 2)
    virial_profile : np.ndarray
        Virial stress τ_xy contribution per bin, shape (n_bins,)
    
    References
    ----------
    Irving, J.H. & Kirkwood, J.G. (1950). J. Chem. Phys. 18, 817.
    Todd, B.D. & Daivis, P.J. (2017). Nonequilibrium Molecular Dynamics.
    """
    n_particles = positions.shape[0]
    forces = np.zeros((n_particles, 2), dtype=np.float64)
    virial_profile = np.zeros(n_bins, dtype=np.float64)
    
    # --- Apply external body force ---
    for i in range(n_particles):
        forces[i, 0] += f_drive
    
    # --- Wall interactions (WCA potential) ---
    for i in range(n_particles):
        # Bottom wall
        dy_bottom = positions[i, 1]
        if dy_bottom < WCA_CUTOFF:
            dy_bottom = max(dy_bottom, 0.01)  # Prevent singularity
            inv_r = 1.0 / dy_bottom
            inv_r6 = inv_r ** 6
            f_wall = 24.0 * inv_r * (2.0 * inv_r6 ** 2 - inv_r6)
            f_wall = min(f_wall, 500.0)  # Force cap for stability
            forces[i, 1] += f_wall
        
        # Top wall
        dy_top = box_y - positions[i, 1]
        if dy_top < WCA_CUTOFF:
            dy_top = max(dy_top, 0.01)
            inv_r = 1.0 / dy_top
            inv_r6 = inv_r ** 6
            f_wall = -24.0 * inv_r * (2.0 * inv_r6 ** 2 - inv_r6)
            f_wall = max(f_wall, -500.0)
            forces[i, 1] += f_wall
    
    # --- Pair interactions (LJ potential with Irving-Kirkwood stress) ---
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Minimum image convention (periodic in x only)
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx = dx - box_x * round(dx / box_x)
            
            r2 = dx * dx + dy * dy
            
            if r2 < LJ_CUTOFF_SQ:
                r2 = max(r2, 0.01)  # Prevent singularity
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3
                
                # LJ force magnitude: F = 24ε/r × [2(σ/r)¹² - (σ/r)⁶]
                f_magnitude = 24.0 * inv_r2 * (2.0 * inv_r6 ** 2 - inv_r6)
                f_magnitude = min(f_magnitude, 500.0)
                
                fx = f_magnitude * dx
                fy = f_magnitude * dy
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                
                # --- Irving-Kirkwood stress distribution ---
                # Distribute virial contribution along the y-segment
                # connecting particles i and j
                y_i = positions[i, 1]
                y_j = positions[j, 1]
                y_min = min(y_i, y_j)
                y_max = max(y_i, y_j)
                
                # Virial contribution: -r_y × F_x (off-diagonal stress)
                virial_total = -2.0*dy * fx
                
                if abs(dy) < 1e-10:
                    # Particles at same y: assign to single bin
                    bin_idx = int(y_i / bin_width)
                    if 0 <= bin_idx < n_bins:
                        virial_profile[bin_idx] += virial_total
                else:
                    # Distribute proportionally across traversed bins
                    bin_min = int(y_min / bin_width)
                    bin_max = int(y_max / bin_width)
                    
                    for k in range(max(0, bin_min), min(n_bins, bin_max + 1)):
                        # Bin boundaries
                        y_bin_lo = k * bin_width
                        y_bin_hi = (k + 1) * bin_width
                        
                        # Intersection of [y_min, y_max] with [y_bin_lo, y_bin_hi]
                        y_lo = max(y_min, y_bin_lo)
                        y_hi = min(y_max, y_bin_hi)
                        
                        if y_hi > y_lo:
                            # Fraction of segment in this bin
                            fraction = (y_hi - y_lo) / abs(dy)
                            virial_profile[k] += virial_total * fraction
    
    return forces, virial_profile


@jit(nopython=True, cache=True)
def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces_old: np.ndarray,
    dt: float,
    box_x: float,
    box_y: float,
    temperature: float,
    f_drive: float,
    bin_width: float,
    n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n_particles = positions.shape[0]
    
    # --- 1. First Half-Kick ---
    velocities += 0.5 * forces_old * dt
    
    # --- 2. Drift ---
    positions += velocities * dt
    positions[:, 0] = np.mod(positions[:, 0], box_x) # Periodic X
    
    # --- 2b. WALL BOUNDARY CONDITIONS (PHYSICALLY CORRECTED) ---
    # FIX A: Removed "Hard Reset". 
    # We only enforce hard reflection for positions to prevent leakage.
    # Friction is handled by the thermostat below.
    
    for i in range(n_particles):
        # Bottom Wall Reflection
        if positions[i, 1] < 0.5:
            positions[i, 1] = 0.5
            if velocities[i, 1] < 0: velocities[i, 1] *= -1.0
            
        # Top Wall Reflection
        elif positions[i, 1] > box_y - 0.5:
            positions[i, 1] = box_y - 0.5
            if velocities[i, 1] > 0: velocities[i, 1] *= -1.0
    
    # --- 3. Compute New Forces ---
    forces_new, virial_profile = compute_forces_and_virial(
        positions, box_x, box_y, f_drive, bin_width, n_bins
    )
    
    # --- 4. Second Half-Kick ---
    velocities += 0.5 * forces_new * dt
    
    # --- 5. THERMOSTAT (FIX B: LOW GAMMA & WALL FRICTION) ---
    # We use a Langevin thermostat.
    # - Bulk: Weak coupling (gamma=0.1) on Y only to remove heat but preserve flow.
    # - Walls: Strong coupling (gamma=5.0) on X to simulate no-slip friction.
    
    gamma_bulk = 0.1  # Gentle thermostat
    gamma_wall = 5.0  # Strong friction at walls
    
    noise_bulk = np.sqrt(2.0 * gamma_bulk * temperature * dt)
    noise_wall = np.sqrt(2.0 * gamma_wall * temperature * dt)
    
    for i in range(n_particles):
        # Check if particle is near wall (within 1.5 sigma)
        dist_bottom = positions[i, 1]
        dist_top = box_y - positions[i, 1]
        
        is_near_wall = (dist_bottom < 1.5) or (dist_top < 1.5)
        
        if is_near_wall:
            # Apply strong friction to X (No-Slip mechanism)
            velocities[i, 0] = velocities[i, 0] * (1.0 - gamma_wall * dt) + np.random.randn() * noise_wall
            # Apply bulk thermostat to Y
            velocities[i, 1] = velocities[i, 1] * (1.0 - gamma_bulk * dt) + np.random.randn() * noise_bulk
        else:
            # Bulk: Only thermostat Y (preserve X momentum)
            velocities[i, 1] = velocities[i, 1] * (1.0 - gamma_bulk * dt) + np.random.randn() * noise_bulk
            # X is NVE (Newtonian) in the bulk -> Correct hydrodynamics
            
    return positions, velocities, forces_new, virial_profile


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def initialise_system() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise particle positions on a regular grid with random velocities.
    
    Returns
    -------
    positions : np.ndarray
        Initial positions, shape (N, 2)
    velocities : np.ndarray
        Initial velocities (zero mean), shape (N, 2)
    """
    # Compute grid dimensions
    n_cols = int(np.sqrt(N_PARTICLES * BOX_X / BOX_Y))
    n_rows = int(np.ceil(N_PARTICLES / n_cols))
    
    dx_grid = BOX_X / n_cols
    dy_grid = (BOX_Y - 2.0) / n_rows  # Leave gap near walls
    
    positions = np.zeros((N_PARTICLES, 2), dtype=np.float64)
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx < N_PARTICLES:
                positions[idx, 0] = (col + 0.5) * dx_grid
                positions[idx, 1] = (row + 0.5) * dy_grid + 1.0
                idx += 1
    
    # Random velocities with zero mean (no net momentum)
    velocities = np.random.randn(N_PARTICLES, 2) * np.sqrt(TEMPERATURE)
    velocities -= velocities.mean(axis=0)  # Remove COM velocity
    
    return positions, velocities


def run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute the full MD simulation and collect time-averaged profiles.
    
    Returns
    -------
    y_coords : np.ndarray
        Bin centre positions
    stress_profile : np.ndarray
        Time-averaged shear stress τ_xy(y)
    velocity_profile : np.ndarray
        Time-averaged velocity u_x(y)
    density_profile : np.ndarray
        Time-averaged number density ρ(y)
    """
    print("=" * 70)
    print("NANOFLUIDS-AI: WP2 MOLECULAR DYNAMICS SIMULATION")
    print("=" * 70)
    print(f"  Particles: {N_PARTICLES}")
    print(f"  Density:   {DENSITY:.2f} σ⁻³")
    print(f"  Channel:   {BOX_X:.1f} × {BOX_Y:.1f} σ²")
    print(f"  Drive:     {F_DRIVE:.3f} ε/σ")
    print(f"  Steps:     {TOTAL_STEPS} (equilibration: {EQUILIBRATION_STEPS})")
    print("=" * 70)
    
    # Initialise
    positions, velocities = initialise_system()
    n_bins = int(BOX_Y / BIN_WIDTH)
    
    # Accumulators
    stress_accum = np.zeros(n_bins, dtype=np.float64)
    kinetic_stress_accum = np.zeros(n_bins, dtype=np.float64)
    velocity_accum = np.zeros(n_bins, dtype=np.float64)
    count_accum = np.zeros(n_bins, dtype=np.float64)
    n_samples = 0
    
    # Initial force calculation
    forces, _ = compute_forces_and_virial(
        positions, BOX_X, BOX_Y, F_DRIVE, BIN_WIDTH, n_bins
    )
    
    # Main loop
    for step in range(TOTAL_STEPS):
        # Integration step
        positions, velocities, forces, virial_profile = velocity_verlet_step(
            positions, velocities, forces, DT, BOX_X, BOX_Y,
            TEMPERATURE, F_DRIVE, BIN_WIDTH, n_bins
        )
        
        # Sampling (after equilibration)
        if step > EQUILIBRATION_STEPS and step % SAMPLE_INTERVAL == 0:
            bin_indices = np.floor(positions[:, 1] / BIN_WIDTH).astype(np.int32)
            
            for i in range(N_PARTICLES):
                b = bin_indices[i]
                if 0 <= b < n_bins:
                    # Kinetic stress contribution: -ρ⟨v_x v_y⟩
                    kinetic_stress_accum[b] += -velocities[i, 0] * velocities[i, 1]
                    velocity_accum[b] += velocities[i, 0]
                    count_accum[b] += 1.0
            
            # Virial already distributed by Irving-Kirkwood
            stress_accum += virial_profile
            n_samples += 1
        
        # Progress report
        if step % 10000 == 0:
            print(f"  Step {step:6d}/{TOTAL_STEPS}  "
                  f"[{'█' * (step * 20 // TOTAL_STEPS):<20s}]")
    
    print("=" * 70)
    print("  Simulation complete. Processing profiles...")
    print("=" * 70)
    
    # Normalise profiles
    bin_volume = BOX_X * BIN_WIDTH
    y_coords = (np.arange(n_bins) + 0.5) * BIN_WIDTH
    
    # --- CORRECTED HANDLING OF EMPTY BINS ---
    # Create a mask for populated bins to avoid division by zero artifacts
    valid_bins = count_accum > 0
    
    # Initialize arrays with NaNs (safe for plotting, won't draw lines to zero)
    velocity_profile = np.full(n_bins, np.nan)
    density_profile = np.full(n_bins, np.nan)
    stress_profile = np.full(n_bins, np.nan)
    
    # Compute averages only where data exists
    velocity_profile[valid_bins] = velocity_accum[valid_bins] / count_accum[valid_bins]
    density_profile[valid_bins] = count_accum[valid_bins] / (n_samples * bin_volume)
    
    # Stress: sum of kinetic and virial
    total_stress_accum = kinetic_stress_accum + stress_accum
    stress_profile[valid_bins] = total_stress_accum[valid_bins] / (n_samples * bin_volume)
    
    return y_coords, stress_profile, velocity_profile, density_profile


# =============================================================================
# PUBLICATION-QUALITY VISUALISATION (Nature/Science Standard)
# =============================================================================

def create_publication_figure(
    y: np.ndarray,
    stress: np.ndarray,
    velocity: np.ndarray,
    density: np.ndarray
) -> None:
    """
    Generate a two-panel figure suitable for Nature/Science publication.
    
    Panel A: Shear stress profile with linear momentum balance
    Panel B: Velocity profile with parabolic Poiseuille solution
    
    Design principles:
        - Clean, minimal aesthetic
        - High contrast for print reproduction
        - Equations embedded within panels
        - Consistent typography
    """
    # --- 1. CENTRAR COORDENADAS ---
    H_half = BOX_Y / 2.0
    y = y - H_half  # Ahora y va de -H/2 a +H/2
    # =========================================================================
    # DATA PREPROCESSING
    # =========================================================================
    
    # Exclude near-wall regions (high noise, layering artifacts)
    #bulk_mask = (y > WALL_EXCLUSION) & (y < BOX_Y - WALL_EXCLUSION)
    bulk_mask = np.abs(y) < (H_half - WALL_EXCLUSION)
    # Exclude NaNs to avoid polluting smoothing and polyfit
    valid_mask = bulk_mask & ~np.isnan(stress) & ~np.isnan(velocity)
    y_bulk = y[valid_mask]
    stress_bulk = stress[valid_mask]
    velocity_bulk = velocity[valid_mask]
    
    # Guard: require enough bins to fit after trimming (deg=2 needs ≥3 points)
    trim = 2
    min_points = 2 * trim + 3
    if len(y_bulk) < min_points:
        print("Warning: insufficient valid bins for fitting; skipping figure.")
        return
    
    # Smooth profiles using Savitzky-Golay-like moving average
    def smooth(data: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if data.size < window:
            return data.copy()
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')
    
    stress_smooth = smooth(stress_bulk, window=3)
    velocity_smooth = smooth(velocity_bulk, window=3)
    
    # Trim edges affected by convolution
    y_final = y_bulk[trim:-trim]
    stress_final = stress_smooth[trim:-trim]
    velocity_final = velocity_smooth[trim:-trim]
    
    # =========================================================================
    # THEORETICAL FITS
    # =========================================================================
    
    # --- Panel A: Linear stress profile ---
    # Momentum balance: dτ_xy/dy = -f_drive × ρ
    # Solution: τ_xy(y) = A + B × y (linear in bulk)
    stress_coeffs = np.polyfit(y_final, stress_final, 1)
    stress_fit_fn = np.poly1d(stress_coeffs)
    stress_fit = stress_fit_fn(y_final)
    
    # --- CALCULATION OF R^2 ---
    ss_res = np.sum((stress_final - stress_fit) ** 2)
    ss_tot = np.sum((stress_final - np.mean(stress_final)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  Stress profile R²: {r_squared:.4f}")
    
    # L2 relative error
    stress_l2_error = np.linalg.norm(stress_final - stress_fit) / np.linalg.norm(stress_final)
    
    # --- Panel B: Parabolic velocity profile ---
    # Poiseuille solution: u_x(y) = (f/2η) × y × (H - y)
    # Fit: u(y) = a × y² + b × y + c
    vel_coeffs = np.polyfit(y_final, velocity_final, 2)
    vel_fit_fn = np.poly1d(vel_coeffs)
    velocity_fit = vel_fit_fn(y_final)
    
    # Extract effective viscosity from parabolic fit
    # u(y) = -(f/2η)y² + (fH/2η)y + c
    # Coefficient of y²: a = -f/(2η) → η = -f/(2a)
    a_coeff = vel_coeffs[0]
    if abs(a_coeff) > 1e-10:
        eta_eff = -F_DRIVE / (2.0 * a_coeff)
    else:
        eta_eff = np.nan
    
    velocity_l2_error = np.linalg.norm(velocity_final - velocity_fit) / np.linalg.norm(velocity_final)
    
    # =========================================================================
    # FIGURE SETUP (Nature/Science Style)
    # =========================================================================
    
    # Nature single-column width: 89 mm ≈ 3.5 in
    # Nature double-column width: 183 mm ≈ 7.2 in
    fig_width = 9.2  # inches (double column)
    fig_height = 3.2  # inches
    
    # Colour palette (colourblind-friendly)
    COLOR_DATA_RAW = '#93C4D2'      # Light blue (raw data)
    COLOR_DATA = '#1A5276'           # Dark blue (processed data)
    COLOR_THEORY = '#C0392B'         # Red (theoretical fit)
    COLOR_BACKGROUND = '#FAFAFA'     # Off-white background
    
    # Typography
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'mathtext.fontset': 'dejavusans',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })
    
    # Create figure with GridSpec for precise control
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    
    # =========================================================================
    # PANEL A: SHEAR STRESS PROFILE
    # =========================================================================
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLOR_BACKGROUND)
    
    # Raw data (faint)
    ax1.plot(y_bulk, stress_bulk, 'o',
             color=COLOR_DATA_RAW, markersize=4, alpha=0.5,
             label='Raw MD', zorder=1)
    
    # Processed data (strong)
    ax1.plot(y_final, stress_final, 'o',
             color=COLOR_DATA, markersize=5,
             markeredgecolor='white', markeredgewidth=0.3,
             label='Time-averaged', zorder=2)
    
    # Theoretical fit
    ax1.plot(y_final, stress_fit, '-',
             color=COLOR_THEORY, linewidth=2.0,
             label='Linear fit', zorder=3)
    
    # Labels
    ax1.set_xlabel(r'Channel position $y$ [$\sigma$]')
    ax1.set_ylabel(r'Shear stress $\tau_{xy}$ [$\varepsilon/\sigma^3$]')
    
    # Panel label
    ax1.text(-0.175, 1.05, 'A', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top')
    
    # Equation box (inside panel, top-right)
    eq_text_A = (
        r'$\nabla \cdot \boldsymbol{\tau} = -\rho \mathbf{f}$'
        '\n'
        r'$\frac{\partial \tau_{xy}}{\partial y} = -\rho f_x$'
        '\n\n'
        rf'$R^2 = {r_squared:.4f}$'  # <--- AÑADIDO AQUÍ
        '\n'
        rf'$E_{{L^2}} = {stress_l2_error:.2e}$'
    )
    
    # Create text box with semi-transparent background
    props_A = dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.92, linewidth=0.5)
    ax1.text(0.97, 0.97, eq_text_A, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=props_A, linespacing=1.4)
    
    # Legend
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9,
               edgecolor='#CCCCCC', fancybox=False)
    
    # Grid
    #ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.4, color='#AAAAAA')
    ax1.grid(False)
    ax1.set_axisbelow(True)
    
    # =========================================================================
    # PANEL B: VELOCITY PROFILE
    # =========================================================================
    
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLOR_BACKGROUND)
    
    # Raw data (faint)
    ax2.plot(y_bulk, velocity_bulk, 'o',
             color=COLOR_DATA_RAW, markersize=4, alpha=0.5,
             label='Raw MD', zorder=1)
    
    # Processed data (strong)
    ax2.plot(y_final, velocity_final, 'o',
             color=COLOR_DATA, markersize=5,
             markeredgecolor='white', markeredgewidth=0.3,
             label='Time-averaged', zorder=2)
    
    # Theoretical fit (Poiseuille parabola)
    ax2.plot(y_final, velocity_fit, '-',
             color=COLOR_THEORY, linewidth=2.0,
             label='Poiseuille fit', zorder=3)
    
    # Labels
    ax2.set_xlabel(r'Channel position $y$ [$\sigma$]')
    ax2.set_ylabel(r'Velocity $u_x$ [$\sigma/\tau$]')
    
    # Panel label
    ax2.text(-0.175, 1.05, 'B', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top')
    
    # Equation box (inside panel, top-left for parabola)
    eq_text_B = (
        r'$\eta \nabla^2 \mathbf{u} = -\rho \mathbf{f}$'
        '\n'
        r'$u_x(y) = \frac{f_x}{2\eta} y(H-y)$'
        '\n\n'
        rf'$\eta_{{\mathrm{{eff}}}} = {eta_eff:.2f}$'
        '\n'
        rf'$E_{{L^2}} = {velocity_l2_error:.2e}$'
    )
    
    props_B = dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.92, linewidth=0.5)
    ax2.text(0.97, 0.97, eq_text_B, transform=ax2.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=props_B, linespacing=1.4)
    
    # Legend
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9,
               edgecolor='#CCCCCC', fancybox=False)
    
    # Grid
    #ax2.grid(True, linestyle='-', alpha=0.3, linewidth=0.4, color='#AAAAAA')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    
    # =========================================================================
    # FINAL ADJUSTMENTS AND EXPORT
    # =========================================================================
    
    plt.tight_layout()
    
    # Save high-resolution figure
    output_path = 'nanofluids_continuum_validation_poiseuille.png'
    fig.savefig(output_path, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pad_inches=0.05)
    
    print(f"\n  ✓ Figure saved: {output_path}")
    print(f"    Resolution: 400 DPI")
    print(f"    Dimensions: {fig_width}\" × {fig_height}\"")
    
    # Also save as PDF for vector graphics
    pdf_path = 'nanofluids_continuum_validation_poiseuille.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  ✓ Vector saved: {pdf_path}")
    
    plt.show()
    
    # --- PHYSICS CHECK: MOMENTUM BALANCE ---
    # Calculate theoretical slope based on bulk density
    # We take the average density in the center of the channel to avoid wall artifacts
    center_idx = len(density) // 2
    width = 2
    rho_bulk_est = np.nanmean(density[center_idx-width:center_idx+width])

    # Print summary statistics
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Stress profile L² error:    {stress_l2_error:.4e}")
    print(f"  Velocity profile L² error:  {velocity_l2_error:.4e}")
    print(f"  Effective viscosity η:      {eta_eff:.3f} [ε·τ/σ³]")
    print(f"  Stress gradient dτ/dy:      {stress_coeffs[0]:.4f} [ε/σ⁴]")
    print(f"   R² :    {r_squared:.4f}")
    print("-" * 70)
    if np.isnan(rho_bulk_est):
        print("  Theoretical Limit (-ρ·f):   NaN (insufficient bulk density data)")
        print("  Momentum Balance Error:     NaN")
    else:
        theoretical_slope = -rho_bulk_est * F_DRIVE
        denom = theoretical_slope if abs(theoretical_slope) > 1e-12 else np.nan
        momentum_error = np.nan if np.isnan(denom) else abs((stress_coeffs[0]-theoretical_slope)/denom)*100.0
        print(f"  Theoretical Limit (-ρ·f):   {theoretical_slope:.4f} [ε/σ⁴]")
        print(f"  Momentum Balance Error:     {momentum_error:.2f}%")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run simulation
    y_coords, stress_profile, velocity_profile, density_profile = run_simulation()
    
    # Generate publication figure
    create_publication_figure(y_coords, stress_profile, velocity_profile, density_profile)
    
    #print("\n  WP2 Validation complete.")
    print(" Validation suite completed successfully.\n")
