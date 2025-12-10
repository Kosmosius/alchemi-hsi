# Section 4: Data and metadata model (ALCHEMI)

This project uses the Section‑4 canonical data model across adapters and physics
utilities. The key types are :class:`alchemi.spectral.sample.Sample`,
:class:`alchemi.types.Spectrum`, :class:`alchemi.spectral.sample.BandMetadata`,
and SRF helpers in :mod:`alchemi.spectral.srf`.

## Minimal end-to-end example

```python
from alchemi.data.adapters.emit import iter_emit_pixels
from alchemi.physics.rad_reflectance import radiance_sample_to_toa_reflectance

# Load a tiny EMIT fixture or user-provided scene (L1B radiance product)
samples = list(iter_emit_pixels("/path/to/emit_l1b_fixture.nc", include_quality=True))
sample = samples[0]

print(sample.spectrum.kind)           # QuantityKind.RADIANCE
print(sample.spectrum.units)          # ValueUnits.RADIANCE_W_M2_SR_NM
print(sample.band_meta.center_nm[:5]) # wavelength grid (nm)
print(sample.quality_masks.keys())    # e.g., {'valid_band', 'deep_water_vapour', ...}

# Convert to TOA reflectance using viewing geometry carried by the sample
reflectance_sample = radiance_sample_to_toa_reflectance(sample)
print(reflectance_sample.spectrum.kind)   # QuantityKind.REFLECTANCE
print(reflectance_sample.spectrum.units)  # ValueUnits.REFLECTANCE_FRACTION
```

A similar flow works for EnMAP fixtures using :func:`iter_enmap_pixels`.

## Core invariants

* Wavelength grids are stored in **nanometres**, strictly increasing.
* Radiance values use W·m⁻²·sr⁻¹·nm⁻¹; reflectance is unitless fraction; brightness
  temperature uses Kelvin.
* SRF matrices are normalised per band so a flat spectrum remains flat after
  convolution; SRF provenance (official/gaussian/none) is recorded in
  ``band_meta.srf_source`` and the optional ``SensorSRF.meta``.
* ``quality_masks`` follow consistent naming (``valid_band``,
  ``deep_water_vapour``, mission-specific detector masks) and always match the
  spectrum length.
