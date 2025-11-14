# MAKO COMEX ingestion quickstart

The COMEX campaign distributes MAKO longwave infrared (LWIR) products as ENVI
``.dat`` / ``.hdr`` pairs. This quickstart walks through downloading a small
flightline, loading the Level-2S radiance, Level-3 brightness temperature (BT),
and Level-3 atmospheric characterization experiment (ACE) scores, and inspecting
basic metadata. For background on band counts, units, and processing levels, see
the [DATA_SPEC](DATA_SPEC.md#sensor-products) entry for MAKO. Sensor response
functions are described in [`alchemi.srf.mako`](../src/alchemi/srf/mako.py).

## Directory layout and filenames

COMEX keeps each flightline in its own directory with subfolders for derived
products:

```
COMEX_MAKO_20140924T221530/
├── L1/
│   └── COMEX_MAKO_20140924T221530_L1_radiance.dat + .hdr
├── L2S/
│   └── COMEX_MAKO_20140924T221530_L2S_radiance.dat + .hdr
├── L3/
│   ├── COMEX_MAKO_20140924T221530_L3_btemp.dat + .hdr
│   ├── COMEX_MAKO_20140924T221530_L3_ace.dat + .hdr
│   ├── quicklooks/COMEX_MAKO_20140924T221530_L3_btemp.png
│   └── vector/COMEX_MAKO_20140924T221530_L3_ace_shapes.zip
└── ancillary/ ...
```

Level-1 is the calibrated radiance on the native whiskbroom timeline. Level-2S
re-grids the scanlines to a spatial cube and keeps radiance in microflick
(µW·cm⁻²·sr⁻¹·µm⁻¹). Level-3 includes both brightness temperatures (stored in
°C) and ACE detection scores. The vector shapefiles bundle ACE exceedances, and
PNG quicklooks provide browse imagery for BT and ACE layers.

Download a sample flightline from the [COMEX archive](https://avirisng.jpl.nasa.gov/comex/):

```bash
#! curl -O https://avirisng.jpl.nasa.gov/comex/data/mako/2014/COMEX_MAKO_20140924T221530_L2S_radiance.dat
#! curl -O https://avirisng.jpl.nasa.gov/comex/data/mako/2014/COMEX_MAKO_20140924T221530_L2S_radiance.hdr
```

Repeat for the ``L3`` BT and ACE rasters, and the shapefile / quicklook bundles
if you need vector overlays.

## Load COMEX cubes with `alchemi`

```python
from pathlib import Path

from alchemi.ingest.mako import open_mako_ace, open_mako_btemp, open_mako_l2s

root = Path("/data/comex/COMEX_MAKO_20140924T221530")
l2s_path = root / "L2S" / "COMEX_MAKO_20140924T221530_L2S_radiance.dat"
l3_bt_path = root / "L3" / "COMEX_MAKO_20140924T221530_L3_btemp.dat"
l3_ace_path = root / "L3" / "COMEX_MAKO_20140924T221530_L3_ace.dat"

l2s = open_mako_l2s(l2s_path)
print(l2s)
print(l2s["radiance"].attrs)

l3_bt = open_mako_btemp(l3_bt_path)
print(l3_bt["bt"].attrs["units"], "minimum", float(l3_bt["bt"].min()))

l3_ace = open_mako_ace(l3_ace_path)
print(l3_ace["ace"].coords["gas_name"].values)
```

The helper functions automatically locate the companion ``.hdr`` files. Radiance
is converted to the canonical ``W·m⁻²·sr⁻¹·nm⁻¹`` units while brightness
temperatures are converted to Kelvin.

## Inspect wavelengths and BT histogram

```python
import matplotlib.pyplot as plt
import numpy as np

wavelengths_nm = l2s.coords["wavelength_nm"].values
print("First five wavelengths (nm):", wavelengths_nm[:5])

_, ax = plt.subplots(figsize=(6, 3))
ax.plot(wavelengths_nm, marker="o", linestyle="-")
ax.set_xlabel("Band index")
ax.set_ylabel("Wavelength (nm)")
ax.set_title("Mako spectral grid")
plt.show()

bt_values = l3_bt["bt"].isel(band=32).values.ravel()
_, ax = plt.subplots(figsize=(6, 3))
ax.hist(bt_values, bins=np.linspace(bt_values.min(), bt_values.max(), 40), color="#ff7f0e")
ax.set_xlabel("Brightness temperature (K)")
ax.set_ylabel("Pixel count")
ax.set_title("Band 33 brightness temperature distribution")
plt.show()
```

The Level-3 ACE cube exposes five gas-specific scores with named coordinates:

```python
ace_scores = l3_ace["ace"].sel(gas_name="NH3").values
print("NH3 ACE mean:", float(ace_scores.mean()))
```

Pair the raster with the ``vector/`` shapefile overlays to visualize exceedance
polygons, or use the ``quicklooks/`` PNGs for a quick sanity check.

## Optional: SRF-aware resampling

When comparing to physics simulations or other sensors, reuse the MAKO spectral
response functions to project high-resolution spectra onto MAKO bands:

```python
import numpy as np

from alchemi.srf.mako import build_mako_srf_from_header
from alchemi.srf.resample import project_to_sensor

mako_srf = build_mako_srf_from_header(wavelengths_nm)

highres_wavelengths = np.linspace(7400.0, 13600.0, 512)
highres_radiance = np.interp(highres_wavelengths, wavelengths_nm, l2s["radiance"].isel(y=0, x=0).values)

projected = project_to_sensor(
    highres_wavelengths,
    highres_radiance,
    mako_srf.centers_nm,
    srf=mako_srf,
)
print("Projected spectrum shape:", projected.shape)
```

The resulting ``projected`` array aligns with MAKO's 128-band grid and can be
compared directly against pixels extracted from the Level-2S cube.
