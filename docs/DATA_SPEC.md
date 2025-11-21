# DATA_SPEC

## Canonical hyperspectral cube

Alchemi normalises all sensor products (SWIR radiance/reflectance and LWIR
radiance/brightness temperature) into a single hyperspectral cube layout. The
cube is represented as an ``xarray.Dataset`` with explicitly named axes,
coordinates, data variables, and attributes so downstream tooling can rely on a
stable schema.

### Dimensions and coordinates

| Name | Description | dtype | Units | Notes |
| --- | --- | --- | --- | --- |
| ``y`` | Along-track row index | ``int32`` | index or metres | Required.
| ``x`` | Cross-track column index | ``int32`` | index or metres | Required.
| ``band`` | Spectral sample index | ``int32`` | – | Required, aligns with spectroscopic order.
| ``wavelength_nm`` | Spectral axis in wavelength | ``float64`` | nm | Monotonic increasing within a small numerical tolerance; required for SWIR and LWIR radiance products.
| ``wavenumber_cm_1`` | Spectral axis in wavenumber | ``float64`` | cm⁻¹ | Optional convenience coordinate for LWIR BT; strictly decreasing.

Both spectral coordinates may coexist. When both are present, they must describe
the same spectral grid (``wavenumber_cm_1 = 1e7 / wavelength_nm``).

The monotonic check for ``wavelength_nm`` follows
``alchemi.types.WAVELENGTH_GRID_MONOTONICITY_EPS`` (default ``1e-9`` nm) to
ignore sub-nanometer floating-point jitter while still rejecting repeated bands
or genuine decreases.

### Data variables

| Variable | Shape | dtype | Units | Purpose |
| --- | --- | --- | --- | --- |
| ``radiance`` | ``(y, x, band)`` | ``float64`` | ``W·m⁻²·sr⁻¹·nm⁻¹`` | Spectral radiance for SWIR or LWIR products.
| ``reflectance`` | ``(y, x, band)`` | ``float64`` | dimensionless | Hemispherical-directional reflectance (SWIR only).
| ``brightness_temp`` | ``(y, x, band)`` | ``float64`` | ``K`` | Brightness temperature spectra (LWIR only).
| ``band_mask`` | ``(band,)`` | ``bool`` | – | Optional boolean mask of valid spectral samples.
| ``qa`` | ``(y, x)`` or ``(y, x, band)`` | implementation-defined | implementation-defined | Optional sensor QA ancillaries.

At least one of ``radiance``, ``reflectance``, or ``brightness_temp`` must be
present. When a modality is absent, its coordinate/unit pair must also be
omitted.

### Valid unit pairs

| Spectral coordinate | Permitted value variable | Value units | Notes |
| --- | --- | --- | --- |
| ``wavelength_nm`` | ``radiance`` | ``W·m⁻²·sr⁻¹·nm⁻¹`` | Canonical SWIR and LWIR radiance grid.
| ``wavelength_nm`` | ``reflectance`` | dimensionless | Reflectance is always expressed on the native wavelength grid.
| ``wavenumber_cm_1`` | ``brightness_temp`` | ``K`` | Canonical LWIR BT grid; may be supplied alongside ``wavelength_nm``.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| ``sensor`` | ``str`` | Human-readable sensor name (e.g., ``"EMIT"``).
| ``product_level`` | ``str`` | Original data level (e.g., ``"L1B"``, ``"L2S"``).
| ``quantity`` | ``{"radiance","reflectance","brightness_temp"}`` or list[str] | Convenience summary of populated data variables.
| ``radiance_units`` | ``str`` | Always ``"W·m⁻²·sr⁻¹·nm⁻¹"`` when ``radiance`` exists.
| ``reflectance_units`` | ``str`` | ``"1"`` when ``reflectance`` exists.
| ``brightness_temp_units`` | ``str`` | ``"K"`` when ``brightness_temp`` exists.
| ``srf_id`` | ``str`` | Cache key or identifier of the spectral response function used for forward models (``sensor:version:hash`` derived from [`alchemi.types.SRFMatrix`](/alchemi/src/alchemi/types.py)).
| ``srf_version`` | ``str`` | Semantic version of the SRF definition (e.g., ``"v01"``).
| ``source_units`` | ``dict`` | Optional mapping describing original sensor units for traceability.
| ``crs`` | implementation-defined | Coordinate reference system metadata retained from source product when available.

When per-band masks originate from the sensor they should be copied into
``band_mask``. Downstream processing may attach additional annotations, but the
canonical fields above must stay intact.

### JSON-ish schema

```json
{
  "dims": {"y": int, "x": int, "band": int},
  "coords": {
    "y": {"dims": ["y"], "dtype": "int32"},
    "x": {"dims": ["x"], "dtype": "int32"},
    "band": {"dims": ["band"], "dtype": "int32"},
    "wavelength_nm": {"dims": ["band"], "dtype": "float64", "units": "nm"},
    "wavenumber_cm_1": {"dims": ["band"], "dtype": "float64", "units": "cm^-1", "optional": true}
  },
  "data_vars": {
    "radiance": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "W·m^-2·sr^-1·nm^-1", "optional": true},
    "reflectance": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "1", "optional": true},
    "brightness_temp": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "K", "optional": true},
    "band_mask": {"dims": ["band"], "dtype": "bool", "optional": true}
  },
  "attrs": {
    "sensor": "str",
    "product_level": "str",
    "quantity": "str | list[str]",
    "srf_id": "str",
    "srf_version": "str",
    "radiance_units": "str?",
    "reflectance_units": "str?",
    "brightness_temp_units": "str?",
    "source_units": "mapping?",
    "crs": "Any?"
  }
}
```

## Sensor-specific examples

The following micro-examples illustrate how existing readers populate the
canonical cube. Shapes are representative; actual scene sizes will be much
larger.

### EMIT L1B → canonical radiance cube

```python
>>> import xarray as xr
>>> from alchemi.data.io.emit import load_emit_l1b
>>> ds = load_emit_l1b("EMIT_L1B_Radiance.tif")
>>> ds
<xarray.Dataset>
Dimensions:      (y: 2, x: 3, band: 285)
Coordinates:
  * y            (y) int32 0 1
  * x            (x) int32 0 1 2
  * band         (band) int32 0 1 2 ... 282 283 284
    wavelength_nm  (band) float64 350.0 351.0 ... 2498.0 2499.0
Data variables:
    radiance     (y, x, band) float64 ...
    band_mask    (band) bool 1 1 ... 0 0
Attributes:
    sensor: "EMIT"
    product_level: "L1B"
    quantity: "radiance"
    radiance_units: "W·m⁻²·sr⁻¹·nm⁻¹"
    srf_id: "emit:v01:<hash>"
    srf_version: "v01"
    source_units: {"radiance": "µW·cm⁻²·sr⁻¹·µm⁻¹"}
```

The reader [`load_emit_l1b`](/alchemi/src/alchemi/data/io/emit.py)
exposes radiance in canonical SI units, attaches the EMIT SRF cache key returned
by [`emit_srf_matrix`](/alchemi/src/alchemi/srf/emit.py), and propagates the
native band mask.

### HyTES L2 BT → canonical BT cube

```python
>>> from alchemi.data.io.hytes import load_hytes_l1b_bt
>>> ds = load_hytes_l1b_bt("HyTES_L2_BT.nc")
>>> ds
<xarray.Dataset>
Dimensions:          (y: 4, x: 4, band: 256)
Coordinates:
  * y                (y) float64 0.0 1.0 2.0 3.0
  * x                (x) float64 0.0 1.0 2.0 3.0
  * band             (band) int32 0 1 ... 254 255
    wavelength_nm    (band) float64 7.5e+03 7.52e+03 ... 1.2e+04
    wavenumber_cm_1  (band) float64 1.333e+03 ... 833.3
Data variables:
    brightness_temp  (y, x, band) float64 ...
    band_mask        (band) bool True True ... True
Attributes:
    sensor: "HyTES"
    product_level: "L1B"
    quantity: "brightness_temp"
    brightness_temp_units: "K"
    srf_id: "hytes:v01:<hash>"
    srf_version: "v01"
```

[`load_hytes_l1b_bt`](/alchemi/src/alchemi/data/io/hytes.py) standardises
dimensions, converts any provided units to Kelvin, and may populate
``wavenumber_cm_1`` when the spectral grid is available in that form.

### MAKO L2S → canonical LWIR radiance cube

```python
>>> from alchemi.io.mako import open_mako_l2s
>>> ds = open_mako_l2s("COMEX_MAKO_L2S.dat")
>>> ds
<xarray.Dataset>
Dimensions:        (y: 2, x: 2, band: 128)
Coordinates:
  * y              (y) int32 0 1
  * x              (x) int32 0 1
  * band           (band) int32 0 1 ... 126 127
    wavelength_nm  (band) float64 7.5e+03 ... 1.35e+04
Data variables:
    radiance       (y, x, band) float64 ...
    band_mask      (band) bool 1 1 ... 1 1
Attributes:
    sensor: "Mako"
    product_level: "L2S"
    quantity: "radiance"
    radiance_units: "W·m⁻²·sr⁻¹·nm⁻¹"
    srf_id: "mako:v01:<hash>"
    srf_version: "v01"
    source_units: {"radiance": "µW·cm⁻²·sr⁻¹·µm⁻¹"}
```

[`open_mako_l2s`](/alchemi/src/alchemi/io/mako.py) handles the ENVI metadata,
performs unit conversion from microflicks, and attaches the canonical SRF ID as
published by [`get_srf("mako")](/alchemi/src/alchemi/srf/registry.py`).

### MAKO L3 BT → canonical LWIR BT cube

```python
>>> from alchemi.io.mako import open_mako_btemp
>>> ds = open_mako_btemp("COMEX_MAKO_L3_BT.dat").rename(bt="brightness_temp")
>>> ds
<xarray.Dataset>
Dimensions:        (y: 2, x: 2, band: 128)
Coordinates:
  * y              (y) int32 0 1
  * x              (x) int32 0 1
  * band           (band) int32 0 1 ... 126 127
    wavelength_nm  (band) float64 7.5e+03 ... 1.35e+04
    wavenumber_cm_1  (band) float64 1.333e+03 ... 740.7
Data variables:
    brightness_temp  (y, x, band) float64 ...
    band_mask      (band) bool True True ... True
Attributes:
    sensor: "Mako"
    product_level: "L3"
    quantity: "brightness_temp"
    brightness_temp_units: "K"
    srf_id: "mako:v01:<hash>"
    srf_version: "v01"
    source_units: {"brightness_temp": "°C"}
```

[`open_mako_btemp`](/alchemi/src/alchemi/io/mako.py) returns Kelvin-scaled
brightness temperature; the snippet above renames the native ``bt`` variable to
the canonical ``brightness_temp``. Downstream utilities such as
[`mako_pixel_bt`](/alchemi/src/alchemi/io/mako.py) extract per-pixel spectra that
conform to [`alchemi.types.Spectrum`](/alchemi/src/alchemi/types.py) using the
same canonical grid.

These examples demonstrate how Phase‑1 consumers can depend on a single,
self-describing schema across SWIR and LWIR modalities.
