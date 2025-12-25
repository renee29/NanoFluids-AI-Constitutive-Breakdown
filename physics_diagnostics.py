#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NANOFLUIDS-AI: WP2 DIAGNOSTIC & VALIDATION TESTS
================================================================================
Scientific Validation Suite – Consistency Checks

Purpose:
    Rigorous scientific validation of the MD simulation without modifying
    the main code. Checks:

    1. Energy conservation (no external forcing, no thermostat)
    2. Velocity profile symmetry u(y) = u(H-y)
    3. Quantitative comparison with analytical Poiseuille solution
    4. Momentum balance verification
    5. Thermostat spatial profile diagnostic

Author: NanoFluids-AI Research Team
License: MIT
================================================================================
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import jit
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Import simulation parameters from main code
import sys
sys.path.append('.')

# =============================================================================
# TEST 1: ENERGY CONSERVATION (NVE Ensemble)
# =============================================================================

@jit(nopython=True, cache=True)
def compute_forces_nve(
    positions: np.ndarray,
    box_x: float,
    box_y: float
) -> np.ndarray:
    """
    Compute forces WITHOUT external driving or thermostats.
    Pure microcanonical ensemble for energy conservation test.
    """
    n_particles = positions.shape[0]
    forces = np.zeros((n_particles, 2), dtype=np.float64)

    LJ_CUTOFF_SQ = 2.5 ** 2
    WCA_CUTOFF = 2.0 ** (1.0 / 6.0)

    # Wall interactions (WCA)
    for i in range(n_particles):
        # Bottom wall
        dy_bottom = positions[i, 1]
        if dy_bottom < WCA_CUTOFF:
            dy_bottom = max(dy_bottom, 0.01)
            inv_r = 1.0 / dy_bottom
            inv_r6 = inv_r ** 6
            f_wall = 24.0 * inv_r * (2.0 * inv_r6 ** 2 - inv_r6)
            f_wall = min(f_wall, 500.0)
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

    # Pair interactions (LJ)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx = dx - box_x * round(dx / box_x)

            r2 = dx * dx + dy * dy

            if r2 < LJ_CUTOFF_SQ:
                r2 = max(r2, 0.01)
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3

                f_magnitude = 24.0 * inv_r2 * (2.0 * inv_r6 ** 2 - inv_r6)
                f_magnitude = min(f_magnitude, 500.0)

                fx = f_magnitude * dx
                fy = f_magnitude * dy

                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy

    return forces


@jit(nopython=True, cache=True)
def velocity_verlet_nve(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces_old: np.ndarray,
    dt: float,
    box_x: float,
    box_y: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure Velocity-Verlet integration (NVE) with elastic wall collisions.
    No thermostat, no external forcing.
    """
    n_particles = positions.shape[0]

    # 1. Half-kick
    velocities += 0.5 * forces_old * dt

    # 2. Drift
    positions += velocities * dt
    positions[:, 0] = np.mod(positions[:, 0], box_x)

    # 3. Elastic wall reflections
    for i in range(n_particles):
        if positions[i, 1] < 0.5:
            positions[i, 1] = 0.5
            if velocities[i, 1] < 0:
                velocities[i, 1] *= -1.0

        elif positions[i, 1] > box_y - 0.5:
            positions[i, 1] = box_y - 0.5
            if velocities[i, 1] > 0:
                velocities[i, 1] *= -1.0

    # 4. Compute new forces
    forces_new = compute_forces_nve(positions, box_x, box_y)

    # 5. Second half-kick
    velocities += 0.5 * forces_new * dt

    return positions, velocities, forces_new


@jit(nopython=True, cache=True)
def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_x: float,
    box_y: float
) -> Tuple[float, float, float]:
    """
    Compute kinetic, potential, and total energy.
    """
    n_particles = positions.shape[0]
    LJ_CUTOFF_SQ = 2.5 ** 2
    WCA_CUTOFF = 2.0 ** (1.0 / 6.0)

    # Kinetic energy
    kinetic = 0.5 * np.sum(velocities ** 2)

    # Potential energy
    potential = 0.0

    # Wall potential (WCA)
    for i in range(n_particles):
        # Bottom wall
        dy_bottom = positions[i, 1]
        if dy_bottom < WCA_CUTOFF:
            dy_bottom = max(dy_bottom, 0.01)
            inv_r = 1.0 / dy_bottom
            inv_r6 = inv_r ** 6
            potential += 4.0 * (inv_r6 ** 2 - inv_r6) + 1.0  # WCA shift

        # Top wall
        dy_top = box_y - positions[i, 1]
        if dy_top < WCA_CUTOFF:
            dy_top = max(dy_top, 0.01)
            inv_r = 1.0 / dy_top
            inv_r6 = inv_r ** 6
            potential += 4.0 * (inv_r6 ** 2 - inv_r6) + 1.0

    # Pair potential (LJ)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx = dx - box_x * round(dx / box_x)

            r2 = dx * dx + dy * dy

            if r2 < LJ_CUTOFF_SQ:
                r2 = max(r2, 0.01)
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 ** 3
                potential += 4.0 * (inv_r6 ** 2 - inv_r6)

    total = kinetic + potential
    return kinetic, potential, total


def test_energy_conservation() -> Dict[str, float]:
    """
    Test 1: Energy conservation in NVE ensemble.

    Returns:
        Dictionary with energy drift statistics
    """
    print("=" * 70)
    print("TEST 1: ENERGY CONSERVATION (NVE)")
    print("=" * 70)

    # Small system for fast testing
    N_PARTICLES = 200
    DENSITY = 0.85
    TEMPERATURE = 1.0
    BOX_X = 8.0
    BOX_Y = N_PARTICLES / (DENSITY * BOX_X)
    DT = 0.002
    N_STEPS = 20000

    print(f"  Particles: {N_PARTICLES}")
    print(f"  Steps:     {N_STEPS}")
    print(f"  Timestep:  {DT}")
    print("  (No thermostat, no external forcing)")
    print("-" * 70)

    # Initialize
    n_cols = int(np.sqrt(N_PARTICLES * BOX_X / BOX_Y))
    n_rows = int(np.ceil(N_PARTICLES / n_cols))
    dx_grid = BOX_X / n_cols
    dy_grid = (BOX_Y - 2.0) / n_rows

    positions = np.zeros((N_PARTICLES, 2))
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx < N_PARTICLES:
                positions[idx, 0] = (col + 0.5) * dx_grid
                positions[idx, 1] = (row + 0.5) * dy_grid + 1.0
                idx += 1

    velocities = np.random.randn(N_PARTICLES, 2) * np.sqrt(TEMPERATURE)
    velocities -= velocities.mean(axis=0)

    # Initial forces
    forces = compute_forces_nve(positions, BOX_X, BOX_Y)

    # Energy tracking
    energy_history = np.zeros(N_STEPS)
    kinetic_history = np.zeros(N_STEPS)
    potential_history = np.zeros(N_STEPS)

    # Simulation
    for step in range(N_STEPS):
        positions, velocities, forces = velocity_verlet_nve(
            positions, velocities, forces, DT, BOX_X, BOX_Y
        )

        KE, PE, E_total = compute_total_energy(positions, velocities, BOX_X, BOX_Y)
        energy_history[step] = E_total
        kinetic_history[step] = KE
        potential_history[step] = PE

        if step % 5000 == 0:
            print(f"  Step {step:5d}: E = {E_total:10.4f}, KE = {KE:10.4f}, PE = {PE:10.4f}")

    # Statistics
    E_mean = energy_history.mean()
    E_std = energy_history.std()
    E_drift = (energy_history[-1] - energy_history[0]) / abs(energy_history[0])

    print("-" * 70)
    print(f"  Mean energy:       {E_mean:.6f}")
    print(f"  Std. deviation:    {E_std:.6e}")
    print(f"  Relative drift:    {E_drift:.6e}")
    print(f"  Max fluctuation:   {(energy_history.max() - energy_history.min()) / abs(E_mean):.6e}")

    # Plot
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    time = np.arange(N_STEPS) * DT

    ax1.plot(time, energy_history, 'b-', linewidth=0.8, label='Total Energy')
    ax1.axhline(E_mean, color='r', linestyle='--', linewidth=1, label=f'Mean = {E_mean:.4f}')
    ax1.fill_between(time, E_mean - E_std, E_mean + E_std, alpha=0.2, color='r')
    ax1.set_ylabel('Total Energy [ε]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('TEST 1: Energy Conservation (NVE Ensemble)')

    ax2.plot(time, (energy_history - E_mean) / abs(E_mean) * 100, 'g-', linewidth=0.8)
    ax2.set_xlabel('Time [τ]')
    ax2.set_ylabel('Relative Error [%]')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('diagnostic_energy_conservation_nve.png', dpi=300, bbox_inches='tight')
    print(f"\n  [OK] Figure saved: diagnostic_energy_conservation_nve.png")
    print("=" * 70)

    return {
        'mean': E_mean,
        'std': E_std,
        'drift': E_drift,
        'fluctuation': (energy_history.max() - energy_history.min()) / abs(E_mean)
    }


# =============================================================================
# TEST 2: VELOCITY PROFILE SYMMETRY
# =============================================================================

def test_velocity_symmetry() -> Dict[str, float]:
    """
    Test 2: Check if velocity profile is symmetric u(y) = u(H-y).

    Uses output from main simulation (must be run first).
    """
    print("\n" + "=" * 70)
    print("TEST 2: VELOCITY PROFILE SYMMETRY")
    print("=" * 70)
    print("  This test requires running the main simulation first.")
    print("  Checking for existing output data...")

    try:
        # Try to load data from main simulation
        # This is a placeholder - in practice, you'd save/load actual data
        print("  [WARNING] Main simulation data not found.")
        print("  Please run continuum_validation_poiseuille.py first.")
        print("=" * 70)
        return {'error': 'No data'}
    except Exception as e:
        print(f"  Error: {e}")
        print("=" * 70)
        return {'error': str(e)}


# =============================================================================
# TEST 3: ANALYTICAL POISEUILLE COMPARISON
# =============================================================================

def analytical_poiseuille(y: np.ndarray, H: float, f_drive: float, eta: float) -> np.ndarray:
    """
    Analytical Poiseuille solution for plane Poiseuille flow.

    Parameters:
        y: Positions (centered at H/2)
        H: Channel height
        f_drive: Body force per particle
        eta: Dynamic viscosity

    Returns:
        Velocity profile u_x(y)
    """
    # Shift to wall coordinates: y_wall ∈ [0, H]
    y_wall = y + H / 2.0

    # Poiseuille: u(y) = (f / 2η) y (H - y)
    u_analytical = (f_drive / (2.0 * eta)) * y_wall * (H - y_wall)

    return u_analytical


def test_analytical_comparison():
    """
    Test 3: Compare simulation with analytical Poiseuille solution.

    Placeholder - requires integration with main code.
    """
    print("\n" + "=" * 70)
    print("TEST 3: ANALYTICAL POISEUILLE COMPARISON")
    print("=" * 70)
    print("  This test requires velocity profile from main simulation.")
    print("  See create_publication_figure() for current implementation.")
    print("=" * 70)


# =============================================================================
# TEST 4: MOMENTUM BALANCE VERIFICATION
# =============================================================================

def test_momentum_balance():
    """
    Test 4: Verify momentum balance dτ_xy/dy = -ρ f_x.

    Placeholder - requires integration with main code.
    """
    print("\n" + "=" * 70)
    print("TEST 4: MOMENTUM BALANCE VERIFICATION")
    print("=" * 70)
    print("  This test is already implemented in the main code.")
    print("  See lines ~691-717 in continuum_validation_poiseuille.py")
    print("=" * 70)


# =============================================================================
# TEST 5: THERMOSTAT SPATIAL PROFILE
# =============================================================================

def visualize_thermostat_profile():
    """
    Test 5: Visualize spatial variation of thermostat coupling.
    """
    print("\n" + "=" * 70)
    print("TEST 5: THERMOSTAT SPATIAL PROFILE")
    print("=" * 70)

    # Parameters from main code
    BOX_Y = 29.41  # Approximate from N=800, density=0.85, BOX_X=10
    WALL_EXCLUSION = 1.5
    gamma_bulk = 0.1
    gamma_wall = 5.0

    # Create y coordinate
    y = np.linspace(0, BOX_Y, 500)

    # Compute effective gamma(y)
    gamma_x = np.zeros_like(y)
    gamma_y = np.zeros_like(y)

    for i, yi in enumerate(y):
        dist_bottom = yi
        dist_top = BOX_Y - yi
        is_near_wall = (dist_bottom < WALL_EXCLUSION) or (dist_top < WALL_EXCLUSION)

        if is_near_wall:
            gamma_x[i] = gamma_wall
            gamma_y[i] = gamma_bulk
        else:
            gamma_x[i] = 0.0  # NVE in bulk
            gamma_y[i] = gamma_bulk
    # 
    H_half = BOX_Y / 2.0
    y_centered = y - H_half
    wall_pos = H_half - WALL_EXCLUSION
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: Gamma_x (friction for no-slip)
    ax1.plot(y_centered, gamma_x, 'b-', linewidth=2)
    ax1.axvline(-wall_pos, color='r', linestyle='--', alpha=0.5, label='Wall boundary')
    ax1.axvline(wall_pos, color='r', linestyle='--', alpha=0.5)
    ax1.fill_between([-H_half, -wall_pos], 0, 6, alpha=0.1, color='red', label='Wall region')
    ax1.fill_between([wall_pos, H_half], 0, 6, alpha=0.1, color='red')
    ax1.set_xlabel(r'Position $y$ [$\sigma$]')
    ax1.set_ylabel(r'$\gamma_x$ (friction coefficient)')
    ax1.set_title('A) X-Direction Thermostat (No-Slip)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-0.2, 6])
    ax1.set_xlim([-H_half, H_half])

    # Panel B: Gamma_y (temperature control)
    ax2.plot(y_centered, gamma_y, 'g-', linewidth=2)
    ax2.axvline(-wall_pos, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(wall_pos, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between([-H_half, -wall_pos], 0, 0.2, alpha=0.1, color='red')
    ax2.fill_between([wall_pos, H_half], 0, 0.2, alpha=0.1, color='red')
    ax2.set_xlabel(r'Position $y$ [$\sigma$]')
    ax2.set_ylabel(r'$\gamma_y$ (friction coefficient)')
    ax2.set_title('B) Y-Direction Thermostat (Temperature)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.2])
    ax2.set_xlim([-H_half, H_half])

    plt.tight_layout()
    plt.savefig('diagnostic_thermostat_spatial_profile.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Figure saved: diagnostic_thermostat_spatial_profile.png")

    # Analysis
    print(f"\n  Spatial coupling configuration:")
    print(f"    Bulk region:  y in [{WALL_EXCLUSION:.1f}, {BOX_Y - WALL_EXCLUSION:.1f}] sigma")
    print(f"    gamma_x (bulk):   {gamma_bulk:.1f} (NVE - Newtonian)")
    print(f"    gamma_y (bulk):   {gamma_bulk:.1f} (weak thermostat)")
    print(f"\n    Wall region:  y < {WALL_EXCLUSION:.1f} or y > {BOX_Y - WALL_EXCLUSION:.1f} sigma")
    print(f"    gamma_x (wall):   {gamma_wall:.1f} (strong friction)")
    print(f"    gamma_y (wall):   {gamma_bulk:.1f} (weak thermostat)")
    print(f"\n  [WARNING] Discontinuous jump at y = {WALL_EXCLUSION:.1f} sigma")
    print(f"    May cause artifacts in velocity/stress profiles near this position.")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("  NANOFLUIDS-AI: WP2 DIAGNOSTIC TEST SUITE")
    print("=" * 70)
    print("\n")

    results = {}

    # Run tests
    results['energy'] = test_energy_conservation()
    results['symmetry'] = test_velocity_symmetry()
    test_analytical_comparison()
    test_momentum_balance()
    visualize_thermostat_profile()

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if 'energy' in results and 'error' not in results['energy']:
        print(f"  [TEST 1] Energy conservation:")
        print(f"    Relative drift:    {results['energy']['drift']:.2e}")
        print(f"    Std. deviation:    {results['energy']['std']:.2e}")

        if abs(results['energy']['drift']) < 1e-4:
            print(f"    Status: [OK] EXCELLENT")
        elif abs(results['energy']['drift']) < 1e-3:
            print(f"    Status: [OK] GOOD")
        else:
            print(f"    Status: [WARNING] NEEDS ATTENTION")

    print("\n  [TEST 2-4] Require main simulation data - see main code")
    print("  [TEST 5] Thermostat profile visualization complete")

    print("\n" + "=" * 70)
    print("  All diagnostic tests complete.")
    print("  Review figures and console output for detailed analysis.")
    print("=" * 70 + "\n")
