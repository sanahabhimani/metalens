# metalens

Python tools for processing metrology data and fitting optical surfaces during the fabrication of metamaterial lenses and filters.

The `metalens` package supports the Metamaterials Lab fabrication workflow by converting raw metrology measurements into surface models, diagnostic plots, and geometry corrections that can be used to prepare optics for machining. It includes utilities for lens and plane fitting, alumina-optic analysis, test-touch calibration, and general metrology data handling.

## Overview

Fabricating metamaterial anti-reflection structures requires accurate knowledge of the surface geometry of each optic before cuts begin. Small offsets, tilts, curvature differences, or setup variations can affect the depth and uniformity of the final structures.

`metalens` provides analysis tools for:

- fitting measured lens surfaces;
- fitting planes and flanges from metrology scans;
- processing alumina lens and filter measurements;
- analyzing test-touch measurements used to validate machine setup;
- applying coordinate transformations and geometry corrections;
- generating plots for metrology validation and fabrication preparation.

The package is used alongside the Metamaterials Lab machining-control software. While the control software manages motion, probing, and cutting operations, `metalens` focuses on interpreting the measured geometry of the optic and preparing the information needed before machining begins.

## Repository Structure

```text
metalens/
├── alumina.py           # Analysis tools for alumina optics
├── core_utils.py        # Shared utility functions
├── housekeeping.py      # General file, formatting, and workflow helpers
├── lensfit.py           # Lens-surface fitting tools
├── planefit.py          # Plane and flange fitting tools
├── test_touch_utils.py  # Test-touch calibration and validation utilities
├── docs/                # Documentation source files
├── __init__.py
└── README.md
