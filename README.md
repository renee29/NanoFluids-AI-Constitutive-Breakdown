# NanoFluids-AI: Continuum Theory Validation Suite

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Status: Validated](https://img.shields.io/badge/Status-Validated-brightgreen.svg)

## Overview
This repository provides a molecular dynamics validation suite for the hydrodynamic limit. It demonstrates the emergence of the Navier-Stokes equations from discrete particle dynamics using the Irving-Kirkwood stress formalism, and validates the Poiseuille flow solution in a nanoconfined channel.

## Visual Demonstration (Key Result)
<p align="center">
  <img src="multiphysics_publication.gif" alt="Hydrodynamic Limit Convergence" width="800">
</p>
<p align="center"><em>Figure 1: Real-time convergence of discrete molecular dynamics to the analytical Navier-Stokes solution.</em></p>

## Validation Metrics
![Stress and Velocity Profiles](nanofluids_continuum_validation_poiseuille.png)

![Energy Conservation (NVE)](diagnostic_energy_conservation_nve.png)

![Thermostat Profile](diagnostic_thermostat_spatial_profile.png)

## Usage
Main workflow:
```bash
python continuum_validation_poiseuille.py
```

Optional diagnostics:
```bash
python physics_diagnostics.py
```

## Outputs
- `nanofluids_continuum_validation_poiseuille.png` - stress and velocity validation figure.
- `nanofluids_continuum_validation_poiseuille.pdf` - vector version of the validation figure.
- `diagnostic_energy_conservation_nve.png` - energy conservation test.
- `diagnostic_thermostat_spatial_profile.png` - thermostat spatial profile.
- `multiphysics_publication.gif` - multiphysics convergence animation.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Citation
See `CITATION.cff` for citation metadata.

## License
MIT. See `LICENSE`.
