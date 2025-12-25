# Changelog

All notable changes to the NanoFluids-AI WP2 Molecular Dynamics Validation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned Improvements (Post-Proposal)
- Implement neighbour lists (Verlet/Cell) to achieve O(N) scaling.
- Refine energy conservation in NVE by replacing hard-wall reflections with pure WCA potentials.
- Implement smooth thermostat transitions (tanh profile) to remove boundary discontinuities.
- Add GPU acceleration support (CuPy/Numba CUDA).
- Benchmark performance against LAMMPS reference implementations.

---

## [1.0.0-proposal] - 2025-12-06

### Description
Initial release for **ERC Consolidator Grant 2026 proposal** (Part B2, Work Package 2: Continuum Theory Validation). This version demonstrates the feasibility of validating continuum hydrodynamic theory from molecular dynamics simulations.

### Added
- **Main simulation code** (`WP2_Fig2_Validation_LJ_Poiseuille.py`)
  - Nonequilibrium MD with Lennard-Jones potential.
  - Velocity-Verlet integration (dt=0.002 τ_LJ).
  - Irving-Kirkwood stress tensor calculation (bond-line integral).
  - WCA repulsive walls for confinement.
  - Langevin thermostat with spatial coupling (bulk vs. wall).
  - Body-force driving for Poiseuille flow.
  - Publication-quality figure generation (Nature/Science style).

- **Diagnostic test suite** (`WP2_Diagnostic_Tests.py`)
  - Energy conservation test (NVE ensemble).
  - Thermostat spatial profile visualisation.

- **Validation metrics**
  - Linear regression on shear stress profile.
  - Parabolic fitting on velocity profile.
  - R² coefficient of determination.
  - L² relative errors.
  - Effective viscosity extraction.

- **Documentation**
  - Comprehensive README with scientific context.
  - CITATION.cff with professional metadata.
  - LICENSE (MIT).
  - TECHNICAL_ANALYSIS.txt with detailed diagnostics.
  - requirements.txt with pinned dependencies.

- **Output files**
  - WP2_NanoFluidsAI_Validation.png (400 DPI raster).
  - WP2_NanoFluidsAI_Validation.pdf (vector graphics).

### Performance
- Current implementation: O(N²) scaling (Direct pair interaction).
- N=800 particles: ~5-15 minutes on modern CPU.
- Numba JIT compilation: ~10× speedup vs. pure Python.

### Known Limitations
- **Energy drift (~2%)** in NVE ensemble (documented in TECHNICAL_ANALYSIS.txt).
  - Cause: Wall reflections break Velocity-Verlet temporal symmetry.
  - Impact: Acceptable for qualitative validation; negligible in NVT production runs.
  
- **Discontinuous thermostat** at y=1.5σ.
  - May cause minor artefacts in velocity/stress profiles near walls.

- **O(N²) force calculation**.
  - Practical limit: N < 2000 on a single CPU.

### Validation Results (Benchmark)
Metrics obtained from the v1.0.0 release build (N=800, 100k steps):
- **Stress profile linearity (R²):** > 0.99 (Excellent agreement with momentum balance).
- **Stress profile L² error:** < 6%.
- **Velocity profile L² error:** < 4% (Robust recovery of parabolic flow).
- **Effective viscosity:** Successfully extracted from molecular trajectories.

### Technical Details
**Physical parameters** (default):
- N = 800 particles
- ρ = 0.85 σ⁻³ (bulk density)
- T = 1.0 ε/k_B (temperature)
- f_drive = 0.02 ε/σ (body force)
- Channel: 10.0 × 29.4 σ²

**Numerical parameters**:
- Timestep: 0.002 τ_LJ
- Total steps: 100,000
- Equilibration: 5,000 steps
- Sampling interval: every 5 steps
- Bin width: 0.6 σ

### Citation
```bibtex
@software{nanofluids_ai_wp2_2025,
  author    = {Fábregas, René and NanoFluids-AI Research Team},
  title     = {NanoFluids-AI WP2: Molecular Dynamics Validation},
  version   = {1.0.0-proposal},
  year      = {2025},
  month     = {12},
  url       = {https://github.com/renee29/NanoFluids-AI},
  license   = {MIT}
}
