# Physics layer overview

This guide summarises the physics utilities described in Section 5 of the ALCHEMI
spec and shows common entry points. All wavelengths are nanometres unless noted.

## Core capabilities

- **Planck & brightness temperature (LWIR)** – Evaluate Planck's law and invert
  radiance to brightness temperature (BT), optionally incorporating SRFs for
  band-averaged BTs.
- **Radiance ↔ TOA reflectance** – Convert L1B radiance to top-of-atmosphere
  (TOA) reflectance and back using band solar irradiance, Earth–Sun distance,
  and solar zenith.
- **SWIR radiative-transfer regimes** – Tag scenes as `trusted` vs `heavy`
  atmospheres to decide when simplified TOA approximations apply.
- **Continuum removal & band depths** – Build convex-hull or anchor continua and
  compute band depths/areas/asymmetries for absorption features.
- **Resampling & virtual sensors** – Convolve lab spectra with SRFs, interpolate
  to band centres, or draw synthetic Gaussian SRFs for robustness studies.
- **TES roadmap** – Convenience LWIR radiance ↔ BT wrappers and an emissivity
  proxy to plug into mission-specific TES workflows.

## Usage snippets

### Radiance ↔ brightness temperature

```python
import numpy as np
from alchemi.physics import radiance_to_bt, bt_to_radiance

wavelength_nm = 10500.0  # 10.5 µm
radiance = 5.0  # W·m⁻²·sr⁻¹·nm⁻¹

bt_K = radiance_to_bt(wavelength_nm, radiance)
restored_rad = bt_to_radiance(wavelength_nm, bt_K)
```

### L1B radiance → TOA reflectance

```python
import numpy as np
from alchemi.physics import radiance_to_toa_reflectance, esun_for_sample
from alchemi.spectral.sample import Sample
from alchemi.types import QuantityKind, RadianceUnits

# Radiance spectrum on a nanometre grid
sample = Sample(
    wavelengths=np.linspace(400, 2500, 10),
    values=np.full(10, 80.0),
    kind=QuantityKind.RADIANCE,
    units=RadianceUnits.W_M2_SR_NM,
    meta={"solar_zenith_deg": 30.0},
)

# Project Esun to the same bands (SRFs handled inside)
esun_band = esun_for_sample(sample)
reflectance = radiance_to_toa_reflectance(
    sample,
    esun_band=esun_band.values,
    solar_zenith_deg=sample.meta["solar_zenith_deg"],
)
```

### Continuum-removed band depths

```python
import numpy as np
from alchemi.physics import continuum_remove, compute_band_depth, BandDefinition
from alchemi.types import QuantityKind
from alchemi.spectral.sample import Sample

wl = np.linspace(2000, 2500, 20)
refl = np.linspace(0.4, 0.2, 20)  # toy absorption ramp
spectrum = Sample(wavelengths=wl, values=refl, kind=QuantityKind.REFLECTANCE)

continuum, removed = continuum_remove(spectrum)
band = BandDefinition(center_nm=2250, left_nm=2150, right_nm=2350)

band_depth = compute_band_depth(removed, band)
```

### Lab spectrum → sensor bands via SRFs

```python
import numpy as np
from alchemi.physics import convolve_to_bands, generate_gaussian_srf
from alchemi.types import QuantityKind
from alchemi.spectral.sample import Sample

# High-resolution lab spectrum
lab_wl = np.linspace(900, 2500, 500)
lab_vals = np.sin(lab_wl / 2000) * 0.1 + 0.5
lab_spec = Sample(wavelengths=lab_wl, values=lab_vals, kind=QuantityKind.REFLECTANCE)

# Create a synthetic 6-band sensor SRF
config = generate_gaussian_srf(
    centers_nm=[1000, 1250, 1500, 1750, 2000, 2250],
    fwhm_nm=[20, 20, 20, 25, 25, 30],
)
resampled = convolve_to_bands(lab_spec, config)
```

### SWIR regime classification for a scene

```python
from alchemi.physics import classify_swir_regime

regime = classify_swir_regime(
    solar_zenith_deg=50.0,
    view_zenith_deg=20.0,
    pwv_cm=2.0,
    aod_550=0.2,
    has_heavy_cloud_or_haze=False,
)
# regime == SWIRRegime.TRUSTED
```

### LWIR emissivity proxy from radiance

```python
import numpy as np
from alchemi.physics import compute_lwir_emissivity_proxy
from alchemi.types import QuantityKind, RadianceUnits
from alchemi.spectral.sample import Sample

wl = np.linspace(8000, 12000, 50)
rad = np.full(50, 1.5)
radiance_spec = Sample(
    wavelengths=wl,
    values=rad,
    kind=QuantityKind.RADIANCE,
    units=RadianceUnits.W_M2_SR_NM,
)
proxy = compute_lwir_emissivity_proxy(radiance_spec)
```

## See also

- `src/alchemi/physics/__init__.py` exposes the main convenience imports.
- Solar reference data live under `resources/solar/esun_reference.csv`.
- SRF utilities under `src/alchemi/physics/resampling.py` support both sensor
  and synthetic SRFs.
