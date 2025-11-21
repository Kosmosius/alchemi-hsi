# CLI Quickstart

This quickstart demonstrates the command-line workflows that Phase 1 will unlock for reading EMIT pixels, normalizing sensor response functions (SRFs), and computing band-depth metrics. Each section includes an example Typer command together with the underlying Python API that **will exist** once the corresponding ingestion or physics issues are implemented.

> [!TIP]
> Activate your development environment (`source .venv/bin/activate`) and install dependencies with `make setup` before running the commands. You can invoke the CLI either as `python -m alchemi.cli` or, once packaging lands, simply `alchemi`.

## Read an EMIT pixel

Extract a single calibrated pixel from an EMIT Level-1B granule and save it as JSON.

```bash
python -m alchemi.cli ingest emit-pixel \
  --granule data/raw/emit/L1B_20220830T193521.h5 \
  --row 512 \
  --col 2048 \
  --output tmp/emit_pixel.json
```

The command will emit the spectrum on the canonical wavelength grid defined by the ingestion epic. Under the hood it uses the forthcoming reader class:

```python
from alchemi.data.emit import EmitL1BReader  # will exist after epic-ingest
from alchemi.types import Spectrum  # existing helper for structured spectra

reader = EmitL1BReader("data/raw/emit/L1B_20220830T193521.h5")
pixel: Spectrum = reader.read_pixel(row=512, col=2048)
print(pixel.wavelength_grid_nm[:5], pixel.values[:5])
```

## Normalize a sensor response function

Normalize the tabulated SRF for the EMIT sensor to ensure each band integrates to 1.0.

```bash
python -m alchemi.cli srf normalize \
  --sensor emit \
  --input data/srf/emit_v01.csv \
  --output data/srf/emit_v01.normalized.parquet
```

The CLI delegates to the SRF registry that caches band definitions and performs trapezoidal integration on the native grid:

```python
from alchemi.srf.registry import SRFRegistry  # will exist after epic-srf

registry = SRFRegistry(root="data/srf")
emit_srf = registry.get("emit", version="v01")
normalized = emit_srf.normalize()
normalized.to_parquet("data/srf/emit_v01.normalized.parquet")
```

## Compute a band-depth metric

Once the continuum-removal utilities land, you can compute band-depth for a pixel spectrum directly from the CLI.

```bash
python -m alchemi.cli physics band-depth \
  --spectrum tmp/emit_pixel.json \
  --continuum window:2120-2210 \
  --band-center 2165
```

Programmatic access uses the upcoming physics helpers:

```python
from alchemi.physics.continuum import remove_continuum  # will exist after epic-physics
from alchemi.physics.metrics import band_depth  # will exist after epic-physics
from alchemi.types import Spectrum

pixel = Spectrum.from_json_path("tmp/emit_pixel.json")
continuum_removed = remove_continuum(pixel)
depth = band_depth(continuum_removed, center_nm=2165)
print(f"Band depth at 2165 nm: {depth:.4f}")
```

## Next steps

- Browse the [Definition of Done](../DEFINITION_OF_DONE.md) to understand the quality bar for CLI features.
- Add new demos under `docs/quickstarts/` whenever you introduce a new command.
- Try the `notebooks/quickstart.ipynb` notebook for an interactive variant that builds a synthetic cube, inspects spectra, and tokenises bands without external data.
