# Canonical cube and ingest/physics architecture

- Status: Accepted
- Date: 2025-12-04

## Context
ALCHEMI-HSI must support multiple hyperspectral sensors (EMIT, EnMAP, AVIRIS-NG, HyTES, Mako) and laboratory spectra while providing a unified interface for training and evaluation. Handling each sensor or product as a bespoke pipeline would fragment the data model, making cross-sensor modeling, evaluation, and tooling fragile. The project needs canonical spectral types (wavelength grid, spectra, samples), a cube abstraction for scene-like data, and a stable contract for spectral response functions (SRFs) and physics helpers.

## Decision
- Define canonical spectral types in `alchemi.types`:
  - `WavelengthGrid` represents a monotonic wavelength axis in nanometers.
  - `Spectrum` holds spectral values with mask and metadata.
  - `Sample` represents a per-pixel or laboratory sample containing a spectrum and associated metadata.
  - `SRFMatrix` stores per-band spectral response functions with trapezoidal normalization.
- Use `Cube` (in `alchemi.data.cube`) as the canonical container for scene-like data, with utilities for sampling and converting into `Sample` instances.
- Separate ingest from physics concerns:
  - `alchemi.data.io.*` and `alchemi.ingest.*` handle sensor-specific file formats and map them into Cube/Spectrum/Sample while harmonizing units, SRFs, and masks.
  - `alchemi.physics.*` provides sensor-agnostic physics primitives (Planck radiance/brightness temperature, reflectance-radiance conversions, continuum removal, resampling) used by ingest, tokenization, and evaluation.
- Maintain a structured SRF and sensor registry:
  - `alchemi.srf.*` and `alchemi.registry.*` store SRF curves, sensor metadata, and helpers for constructing canonical representations.

## Consequences
### Positive
- A single, well-typed contract for data and spectra across sensors and lab sources.
- Physics helpers are reusable and testable in isolation from ingest.
- Ingest logic remains modular and testable per sensor.
- Adding new sensors or laboratory libraries reuses the same canonical contracts.

### Negative / Tradeoffs
- Layering types, cubes, ingest, and physics introduces conceptual overhead for new contributors.
- Sensor-specific quirks still require bespoke handling within ingest or physics helpers.
- Changes to canonical types can ripple through ingest, registries, and tokenization code.

## Alternatives Considered
- Per-sensor ad hoc data models with bespoke pipelines; rejected because they hinder cross-sensor modeling, complicate shared tooling, and increase maintenance overhead.
- A purely xarray-based approach without explicit Cube/Sample/Spectrum types, relying on conventions; rejected because it weakens explicit contracts and makes interchangeability and validation harder.
- Performing physics calculations in-place within ingest routines instead of separate physics modules; rejected because it reduces reuse, blurs responsibilities, and makes physics harder to test independently.

## References
- [docs/ARCH.md](../ARCH.md)
- [docs/DATA_SPEC.md](../DATA_SPEC.md)
- [docs/PHYSICS_HELPERS.md](../PHYSICS_HELPERS.md)
- [src/alchemi/types.py](../../src/alchemi/types.py)
- [src/alchemi/data/cube.py](../../src/alchemi/data/cube.py)
- [src/alchemi/data/io/](../../src/alchemi/data/io/)
- [src/alchemi/ingest/](../../src/alchemi/ingest/)
- [src/alchemi/physics/](../../src/alchemi/physics/)
- [src/alchemi/srf/](../../src/alchemi/srf/)
- [src/alchemi/registry/](../../src/alchemi/registry/)
