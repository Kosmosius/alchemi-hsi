# CLI contract

The `alchemi` Typer CLI is the public entry point for inspecting datasets, exporting canonical cubes, and running experimental training tooling. The commands below are considered part of the stable CLI surface; their names and flags are expected to remain compatible across minor releases. Commands marked as **experimental** may change while the project is under active development.

## Stable commands

| Command | Stability | Description |
| --- | --- | --- |
| `alchemi --version` / `alchemi about` | Stable | Prints the installed `alchemi` version, project summary, and the list of supported sensors with their canonical quantity kinds. |
| `alchemi data info PATH` | Stable | Sniffs a cube and prints the sensor, quantity kind, shape, band range, and metadata summary. |
| `alchemi data to-canonical PATH [--out npz|zarr|/path/file]` | Stable | Converts a supported sensor product into the canonical `Cube` representation (NPZ/Zarr) with wavelength grid and band mask. |

## Experimental commands

These commands are available for development and may evolve without notice:

- `alchemi validate-data` – YAML-driven dataset validation and SRF checks.
- `alchemi validate-srf` – Validate SRF integrals for a specific sensor.
- `alchemi evaluate` – Evaluation harness for trained models.
- `alchemi pretrain-mae` – Synthetic MAE baseline experiments.
- `alchemi align train` – Mainline CLIP-style alignment trainer.

## Discoverability and sensor support

Use `alchemi about` (or `alchemi --version`) to quickly learn which version is installed and which sensors are supported. The command also lists the canonical quantity kind for each sensor and reiterates the canonical cube contract (sensor-agnostic values, wavelength coordinates, band masks, and metadata in NPZ or Zarr format).

Invoke `alchemi --help` or `alchemi data --help` for detailed flag-level documentation.
