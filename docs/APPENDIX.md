# APPENDIX

Worked numerical examples matching the helper functions in `alchemi.physics` and
`alchemi.types`.

## SRF normalisation (trapezoidal)
Using `SRFMatrix.normalize_trapz` with a single band sampled on a non-uniform
grid:

* Wavelengths: `[900.0, 901.0, 902.0]` nm
* Response: `[0.2, 0.5, 0.3]`
* Trapezoidal area = `0.75` → normalised response =
  `[0.2667, 0.6667, 0.4000]`
* Verifying: `trapz(normalised, nm) = 1.0`

Any non-positive or non-finite area would raise before creating the normalised
matrix.

## Planck BT ↔ radiance (LWIR)
Using `physics.planck.bt_K_to_radiance` and `radiance_to_bt_K` for a typical
LWIR channel (10.5 µm = 10500 nm):

* Input brightness temperature: `300 K`
* Forward conversion: `bt_K_to_radiance([300], [10500]) ≈ 9.79e-03 W·m⁻²·sr⁻¹·nm⁻¹`
* Inversion: `radiance_to_bt_K([9.79e-03], [10500])` returns `≈ 300.0 K`

Internally, the helper converts per-nanometre radiance to per-metre, applies the
Planck inversion in float64, and clips tiny/overflowing exponent terms for
stability.

## Continuum removal and band depth (SWIR)
Using `physics.swir.continuum_remove` and `band_depth` on a 2.12 µm absorption:

* Grid (nm): `[2100, 2110, 2120, 2130, 2140]`
* Reflectance: `[0.40, 0.35, 0.30, 0.33, 0.38]`
* Continuum between 2100–2140 nm: `[0.400, 0.395, 0.390, 0.385, 0.380]`
* Continuum-removed reflectance: `[1.000, 0.886, 0.769, 0.857, 1.000]`
* Band depth at 2120 nm: `1 - 0.769 ≈ 0.231`

The implementation uses linear interpolation between the bracketing shoulders,
clips the continuum to avoid divide-by-zero, and indexes the absorption centre
via `searchsorted` on the wavelength grid.
