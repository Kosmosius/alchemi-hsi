# Data & physics contracts — M1 alignment

This page condenses the "data & physics foundations" rules established in M1. It is the developer-facing reference for the canonical spectrum/sample types, SRF registry invariants, quantity kinds, trusted-regime flags, and the narrow LWIR scope adopted for v1.1.

## Spectrum semantics

`Spectrum` is the canonical per-pixel (or lab) spectral container (`src/alchemi/types.py`) and underpins chip/cube tensors. Key fields and invariants:

- `wavelengths: WavelengthGrid`
  - Always stored in **nanometres** and strictly increasing.
  - Accepts nm/µm/Å inputs; converts to nm during construction.
- `values: np.ndarray`
  - Bands-last layout; trailing dimension must match the wavelength grid length.
  - Optional boolean `mask` aligned to the band dimension.
- `kind: QuantityKind` and `units`
  - Supported kinds: `radiance`, `reflectance`, `brightness_temperature` (alias `BT`).
  - Canonical units per kind:
    - Radiance: `W·m⁻²·sr⁻¹·nm⁻¹`.
    - Reflectance: dimensionless fraction (percent accepted and rescaled).
    - Brightness temperature: kelvin.
  - Units are normalised on construction; radiance provided per-µm is rescaled to per-nm.
- `meta: dict`
  - Free-form extras copied across conversions.

Construction uses `_normalize_quantity_kind` and `alchemi.physics.units.normalize_values_to_canonical` to enforce canonical units. Conversions:

- **Radiance ↔ BT**: `alchemi.physics.planck.radiance_to_brightness_temp` / `brightness_temp_to_radiance` return new `Spectrum` objects with updated `kind`/`units`.
- **Radiance ↔ TOA reflectance**: `alchemi.physics.rad_reflectance.radiance_to_toa_reflectance` and `toa_reflectance_to_radiance` require explicit radiance spectra in per-nm units and reflectance fractions. Adapters should not silently switch kinds.

For cubes and chips, tensors are shaped `(H, W, L)` or `(N, L)` with a shared wavelength grid; APIs should continue to follow `Spectrum` validation rules for wavelengths, units, and masks.

## Sample semantics

`Sample` (`src/alchemi/spectral/sample.py`) packages a `Spectrum` with acquisition and band metadata. Fields:

- `spectrum: Spectrum` — validated wavelengths/values/kind/units.
- `sensor_id: str` — registry key for SRFs and sensor metadata.
- Acquisition metadata: `acquisition_time`, `geo` (lat/lon/elev), `viewing_geometry` (solar/view angles, Earth–Sun distance).
- Band metadata: `BandMetadata(center_nm, width_nm, valid_mask, srf_source)` with length matching the spectrum.
- `srf_matrix: SRFMatrix | None` — preferred when available for convolution/reprojection; validated against spectrum length and band metadata.
- `quality_masks: dict[str, np.ndarray]` — per-band masks (bad bands, saturation, water-vapour windows, etc.).
- `ancillary: dict` — free extras (row/col indices, atmospheric proxies, experiment flags).

Usage:

- Ingest adapters (`src/alchemi/data/adapters/`) emit `Sample` instances or iterables thereof.
- Datasets (`src/alchemi/data/datasets.py`) return samples keyed by `sensor_id` and validated spectrum shapes.
- Alignment, synthetic data, and evaluation utilities expect `Sample`/`Spectrum`, not raw `(wavelengths, values)` tuples.
- For image cubes, `Sample.from_chip` provides a per-pixel view; chip-level samples can wrap `(H, W, L)` tensors plus shared metadata.

## SRF registry and invariants

The SRF registry (`src/alchemi/registry/srfs.py`) loads sensor response functions keyed by `sensor_id`:

- Resources live under `resources/srfs/*.json` (optionally `.npy`/`.npz` mirrors).
- Each entry stores band centres, SRF wavelength grids, responses, optional bad-band masks/windows, and a `srf_source` tag (e.g., `"official"`, `"gaussian"`, `"none"`).
- Bands are normalised individually (trapezoidal integration → unit area) on load; zero-area SRFs are rejected.
- Returned as `SRFMatrix` with consistent `(bands, wavelengths)` alignment for resampling and adapter checks.

Validation and onboarding:

- `tests/registry/test_registry_consistency.py` checks sensor specs against SRF payloads and per-band normalisation.
- To add a new sensor: drop the SRF JSON under `resources/srfs/`, expose it via `alchemi.registry.srfs.get_srf`, ensure `sensor_id` is registered in `registry/sensors.py`, and run the registry tests.
- SRFs are consumed by `alchemi.physics.resampling.convolve_to_bands` for lab→sensor projection and by adapters to sanity-check band counts/wavelength ranges; randomized SRFs for robustness should preserve per-band normalisation.

## Quantity kinds and conversion rules

Supported kinds: radiance, reflectance, brightness temperature. Central utilities live in `alchemi.physics`:

- Radiance/BT conversions: `planck.radiance_to_brightness_temp` and `planck.brightness_temp_to_radiance` expect radiance in `W·m⁻²·sr⁻¹·nm⁻¹` and wavelengths in nm.
- Radiance/reflectance: `rad_reflectance.radiance_to_toa_reflectance` / `toa_reflectance_to_radiance` require per-nm radiance, TOA reflectance fractions, solar zenith in degrees, and band-matched solar irradiance (via `physics.solar.esun_for_sample` or `get_reference_esun`).
- Resampling: `physics.resampling.convolve_to_bands` preserves `kind`/`units`; inputs must already be in canonical units.

Normalisation rules and gotchas:

- Radiance inputs expressed per-µm are rescaled to per-nm during `Spectrum` construction; wavenumber inputs are also supported with explicit conversions.
- Do not silently change `kind`/`units`—always construct a new `Spectrum` (e.g., after Planck or reflectance transforms).
- Reflectance is dimensionless; percent inputs are scaled to fractions. TOA vs surface reflectance: current utilities compute TOA reflectance; surface corrections are out-of-scope for M1.
- BT is derived on demand; store radiance or reflectance and convert when needed.

## SWIR trusted regime and LWIR scope

- **SWIR trusted vs heavy regimes**: `alchemi.physics.rt_regime.classify_swir_regime` tags metadata as `trusted` or `heavy` using solar/view zenith, precipitable water vapour, aerosol optical depth, and cloud/haze flags. Helpers `swir_regime_for_sample`, `swir_regime_for_scene`, and `trusted_swir` attach or check the regime flag (`sample.ancillary["swir_regime"]`). Trusted vs heavy regimes gate evaluation stratification, lab alignment claims, and LoD sweeps.
- **LWIR v1.1 scope**: LWIR operates in “brightness-temperature only” mode. BT spectra (kelvin) are used directly; full TES (temperature–emissivity separation) is future work (see stubs in `physics/tes.py`). Convert BT↔radiance via Planck utilities when needed; do not assume emissivity outputs in v1.1.

## Ingest + physics examples (REPL-ready snippets)

### Example 1 — EMIT pixel → TOA reflectance → continuum removal

```python
import numpy as np
from alchemi.data.io import load_emit_l1b
from alchemi.physics import continuum
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.physics.solar import esun_for_sample, earth_sun_distance_for_sample

# Load EMIT L1B and pick one pixel
emit_ds = load_emit_l1b("EMIT_L1B_Radiance.h5")
pixel_sample = emit_ds[0]  # adapter returns Sample objects

# Convert radiance → TOA reflectance
esun = esun_for_sample(pixel_sample)
d_au = earth_sun_distance_for_sample(pixel_sample)
reflectance = radiance_to_toa_reflectance(
    pixel_sample.spectrum,
    esun_band=np.asarray(esun),
    d_au=d_au,
    solar_zenith_deg=pixel_sample.viewing_geometry.solar_zenith_deg,
)

# Apply continuum removal
cont_removed = continuum.continuum_remove(reflectance)
```

### Example 2 — SPLIB spectrum → SRF projection → sensor-style Sample

```python
from alchemi.data.adapters import load_splib_spectrum
from alchemi.physics.resampling import convolve_to_bands
from alchemi.registry.srfs import get_srf
from alchemi.spectral.sample import Sample

# High-resolution lab spectrum from SPLIB
lab_spec = load_splib_spectrum("resources/splib/labradorite.txt")

# Sensor SRF (e.g., EMIT)
srf = get_srf("emit")

# Convolve lab spectrum to sensor bands
sensor_spec = convolve_to_bands(lab_spec, srf)

# Wrap in a Sample with band metadata and SRF provenance
sample = Sample(
    spectrum=sensor_spec,
    sensor_id="emit",
    band_meta={
        "center_nm": srf.centers_nm,
        "width_nm": srf.centers_nm * 0 + 10.0,  # placeholder widths
        "valid_mask": srf.bad_band_mask if srf.bad_band_mask is not None else np.ones_like(srf.centers_nm, dtype=bool),
        "srf_source": "official",
    },
    srf_matrix=srf,
)
```

## How to add a new sensor (high level)

1. Add SRF payload under `resources/srfs/` and expose via `registry/srfs.get_srf`.
2. Register the sensor metadata (centres, widths, wavelength range) in `registry/sensors.py`.
3. Implement or extend an ingest adapter under `data/adapters/` to emit `Sample`/`Spectrum` in canonical units.
4. Run SRF and ingest tests (e.g., `tests/registry/test_registry_consistency.py`, adapter-specific tests) to confirm invariants.
