# DATA_SPEC

This file captures the runtime contracts enforced by `alchemi.types` and the
current ingest readers. All loaders first map native products into a canonical
hyperspectral `Cube`, from which `Spectrum` and `Sample` objects are derived for
training/eval. SRFs are represented explicitly via `SRFMatrix` so physics and
band embeddings share a consistent basis.

## Core types

### Wavelength grids
* `WavelengthGrid.nm` is an increasing 1‑D array in nanometres. The monotonicity
  check allows tiny backward steps up to `WAVELENGTH_GRID_MONOTONICITY_EPS =
  1e-9 nm`, rejects near-duplicates within `WAVELENGTH_GRID_DUPLICATE_EPS =
  1e-12 nm`, and requires at least one positive step.

### Spectrum
* Fields: `(wavelengths: WavelengthGrid, values: float[B], kind: SpectrumKind,
  units: str, mask: bool[B]|None, meta: dict)`.
* Allowed kinds/units: radiance (`W·m⁻²·sr⁻¹·nm⁻¹`), reflectance (`1`), and
  brightness temperature (`K`). Unexpected units raise a warning but are
  preserved for traceability.
* Validation: radiance must be non‑negative; reflectance ∈ [0, 1 + 1e‑3]; BT must
  be > 0 K (warnings if outside [150, 400] K). When present, the boolean mask
  must align with the wavelength grid.
* `masked()` returns a new `Spectrum` with the mask applied, preserving metadata.

### Sample
* Fields: `(spectrum: Spectrum, meta: SampleMeta|dict)` where `SampleMeta`
  captures sensor ID, `(row, col)` indices, optional timestamp, and arbitrary
  extras.
* `Cube.sample_at` uses the cube’s spectral axis/kind to populate a `Spectrum`
  plus per-pixel metadata (sensor/SRF identifier and cube attributes). The
  returned `meta` is always a plain dict so callers can serialise it directly.

### SRFMatrix
* Fields: `(sensor: str, centers_nm: float[B], bands_nm: list[float[K]],
  bands_resp: list[float[K]], version: str="v1", cache_key: str|None, bad_band_mask: bool[B]|None, bad_band_windows_nm: Sequence[(float, float)]|None)`.
* Each band row must pair `bands_nm[i]` and `bands_resp[i]` with matching 1‑D
  shapes. `centers_nm` length must match the number of band rows.
* `normalize_trapz()` divides each band response by its trapezoidal integral
  (`utils.integrate.np_integrate`), raising if the area is non‑finite or ≤ 0. The
  returned `SRFMatrix` is a fresh instance with normalised rows and copied masks.
* `row_integrals()` computes the per-band integrals without altering the matrix.

## Canonical hyperspectral cube

Alchemi normalises all sensor products (SWIR radiance/reflectance and LWIR
radiance/brightness temperature) into a single hyperspectral cube layout. The
cube is represented as an `xarray.Dataset` with explicitly named axes,
coordinates, data variables, and attributes so downstream tooling can rely on a
stable schema. Current readers emit wavelength coordinates; the optional
`wavenumber_cm1` axis is produced only when callers work directly with `Cube`
objects parameterised in wavenumber.

### Dimensions and coordinates

| Name | Description | dtype | Units | Notes |
| --- | --- | --- | --- | --- |
| `y` | Along-track row index | `int32` | index or metres | Required.
| `x` | Cross-track column index | `int32` | index or metres | Required.
| `band` | Spectral sample index | `int32` | – | Required, aligns with spectroscopic order.
| `wavelength_nm` | Spectral axis in wavelength | `float64` | nm | Monotonic increasing within a small numerical tolerance; required for SWIR and LWIR radiance products.
| `wavenumber_cm1` | Spectral axis in wavenumber | `float64` | cm⁻¹ | Optional convenience coordinate used when constructing `Cube` instances from wavenumber grids.

Both spectral coordinates may coexist. When both are present, they must describe
the same spectral grid (`wavenumber_cm1 = 1e7 / wavelength_nm`).

The monotonic check for `wavelength_nm` follows
`alchemi.types.WAVELENGTH_GRID_MONOTONICITY_EPS` (default `1e-9` nm) to ignore
sub-nanometre floating-point jitter while still rejecting repeated bands or
genuine decreases.

### Data variables

| Variable | Shape | dtype | Units | Purpose |
| --- | --- | --- | --- | --- |
| `radiance` | `(y, x, band)` | `float64` | `W·m⁻²·sr⁻¹·nm⁻¹` | Spectral radiance for SWIR or LWIR products.
| `reflectance` | `(y, x, band)` | `float64` | dimensionless | Hemispherical-directional reflectance (SWIR only).
| `brightness_temp` | `(y, x, band)` | `float64` | `K` | Brightness temperature spectra (LWIR only). `open_mako_btemp` emits this as `bt`; callers can rename to the canonical key when desired.
| `band_mask` | `(band,)` | `bool` | – | Optional boolean mask of valid spectral samples.
| `qa` | `(y, x)` or `(y, x, band)` | implementation-defined | implementation-defined | Optional sensor QA ancillaries.

At least one of `radiance`, `reflectance`, or `brightness_temp` must be present.
When a modality is absent, its coordinate/unit pair must also be omitted.

### Valid unit pairs

| Spectral coordinate | Permitted value variable | Value units | Notes |
| --- | --- | --- | --- |
| `wavelength_nm` | `radiance` | `W·m⁻²·sr⁻¹·nm⁻¹` | Canonical SWIR and LWIR radiance grid.
| `wavelength_nm` | `reflectance` | dimensionless | Reflectance is always expressed on the native wavelength grid.
| `wavenumber_cm1` | `brightness_temp` | `K` | Canonical LWIR BT grid; may be supplied alongside `wavelength_nm`.

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `sensor` | `str` | Human-readable sensor name (e.g., `"EMIT"`).
| `radiance_units` | `str` | `"W·m⁻²·sr⁻¹·nm⁻¹"` when `radiance` exists.
| `reflectance_units` | `str` | `"1"` when `reflectance` exists.
| `brightness_temp_units` | `str` | `"K"` when `brightness_temp` exists (Mako BT cubes expose `bt_units`).
| `source_radiance_units` / `source_bt_units` | `str` | Optional record of the original sensor units before conversion.
| `quantity` | `str` | Optional summary set by some SWIR readers (e.g., EnMAP, AVIRIS‑NG).
| `srf_id` | `str` | Optional cache key/identifier applied by `alchemi.ingest.*` when SRFs are injected.
| `crs` | implementation-defined | Coordinate reference system metadata retained from source products when available.

When per-band masks originate from the sensor they should be copied into
`band_mask`. Downstream processing may attach additional annotations, but the
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
    "wavenumber_cm1": {"dims": ["band"], "dtype": "float64", "units": "cm^-1", "optional": true}
  },
  "data_vars": {
    "radiance": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "W·m^-2·sr^-1·nm^-1", "optional": true},
    "reflectance": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "1", "optional": true},
    "brightness_temp": {"dims": ["y", "x", "band"], "dtype": "float64", "units": "K", "optional": true},
    "band_mask": {"dims": ["band"], "dtype": "bool", "optional": true}
  },
  "attrs": {
    "sensor": "str",
    "quantity": "str?",
    "srf_id": "str?",
    "radiance_units": "str?",
    "reflectance_units": "str?",
    "brightness_temp_units": "str?",
    "source_radiance_units": "str?",
    "source_bt_units": "str?",
    "crs": "Any?"
  }
}
```

## Sensor-specific notes

* **EMIT/EnMAP/AVIRIS‑NG (SWIR)** — native radiance or reflectance is converted
  to SI radiance; band masks/FWHM are preserved when present. SRF identifiers are
  applied when wrapping the xarray datasets through `alchemi.ingest.*` helpers.
* **HyTES (LWIR)** — `ingest.hytes.from_hytes_bt` expects brightness temperature
  with a wavelength grid and stores BT directly (not radiance). The xarray
  reader normalises dims and units but does not emit a wavenumber coordinate.
* **Mako (LWIR)** — radiance (`open_mako_l2s`) and BT (`open_mako_btemp`) readers
  convert from microflicks/°C to SI units. The BT loader emits a `bt` variable
  (Kelvin) that can be renamed to the canonical `brightness_temp` for consistency
  before ingestion.
* **Lab spectra** — synthetic lab grids for alignment are generated via
`LabGridConfig` inside the alignment trainer and stored as 1‑D `Spectrum`
objects; no spatial dimensions are attached.

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
    radiance_units: "W·m⁻²·sr⁻¹·nm⁻¹"
    source_radiance_units: "µW·cm⁻²·sr⁻¹·µm⁻¹"
```

The reader [`load_emit_l1b`](/alchemi/src/alchemi/data/io/emit.py) exposes
radiance in canonical SI units and propagates the native band mask; SRF IDs are
added later when converting the dataset to a `Cube` through
`ingest.emit.from_emit_l1b`.

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
Data variables:
    brightness_temp  (y, x, band) float64 ...
    band_mask        (band) bool True True ... True
Attributes:
    sensor: "HyTES"
    brightness_temp_units: "K"
```

[`load_hytes_l1b_bt`](/alchemi/src/alchemi/data/io/hytes.py) standardises
dimensions and converts any provided units to Kelvin. Wavenumber coordinates are
not emitted by the reader; any axis conversions are handled when wrapping into a
`Cube`.

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
    radiance_units: "W·m⁻²·sr⁻¹·nm⁻¹"
    source_radiance_units: "µW·cm⁻²·sr⁻¹·µm⁻¹"
```

[`open_mako_l2s`](/alchemi/src/alchemi/io/mako.py) handles the ENVI metadata,
performs unit conversion from microflicks, and preserves band masks. SRF IDs are
added when the dataset is lifted into a `Cube` via `ingest.mako.from_mako_l2s`.

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
Data variables:
    brightness_temp  (y, x, band) float64 ...
    band_mask      (band) bool True True ... True
Attributes:
    sensor: "Mako"
    bt_units: "K"
    source_bt_units: "°C"
```

[`open_mako_btemp`](/alchemi/src/alchemi/io/mako.py) returns Kelvin-scaled
brightness temperature; the snippet above renames the native `bt` variable to the
canonical `brightness_temp`. Downstream utilities such as
[`mako_pixel_bt`](/alchemi/src/alchemi/io/mako.py) extract per-pixel spectra that
conform to [`alchemi.types.Spectrum`](/alchemi/src/alchemi/types.py) using the
same canonical grid; SRF tagging occurs when passing through
`ingest.mako.from_mako_l3`.

These examples demonstrate how Phase‑1 consumers can depend on a single,
self-describing schema across SWIR and LWIR modalities.
