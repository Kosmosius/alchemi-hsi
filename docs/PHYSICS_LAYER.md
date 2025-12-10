# Physics layer responsibilities

The `alchemi.physics` package focuses on lightweight spectral utilities that
operate at the top-of-atmosphere (TOA) level:

- radiance ↔ TOA reflectance conversions under single-layer assumptions,
- Planck-law brightness temperature utilities,
- continuum removal and band metrics,
- sensor resampling helpers that respect Section-4 spectral semantics.

What the physics layer deliberately **does not** do:

- full atmospheric correction or surface reflectance retrieval,
- multi-layer radiative transfer or coupling with external RT solvers,
- modelling adjacency effects, BRDF, or multiple scattering terms.

Expected external inputs
------------------------

- Mission-generated L2A surface reflectance when surface properties are needed.
- Ancillary solar geometry (solar zenith, Earth–Sun distance) and irradiance
  tables for TOA conversions.
- Optional atmospheric summaries (e.g., PWV/AOD or `SWIRRegime`) that allow the
  helpers to emit warnings when used outside their trusted approximation regime.
