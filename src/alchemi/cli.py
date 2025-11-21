from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import typer
import xarray as xr
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]

from .data.cube import Cube
from .data.io import load_avirisng_l1b, load_emit_l1b, load_enmap_l1b, load_hytes_l1b_bt
from .data.validators import validate_dataset, validate_srf_dir
from .io.mako import open_mako_ace, open_mako_btemp, open_mako_l2s
from .srf import SRFRegistry
from .train.alignment_trainer import AlignmentTrainer
from .training.seed import seed_everything
from .training.trainer import run_eval, run_pretrain_mae
from .utils.logging import get_logger

_SUPPORTED_SENSORS = (
    "EMIT L1B",
    "EnMAP L1B",
    "AVIRIS-NG L1B",
    "HyTES L1B",
    "Mako ACE",
    "Mako BTEMP",
    "Mako L2S",
)
_CANONICAL_DESC = (
    "Canonical cubes store sensor-agnostic radiance values, wavelength coordinates, "
    "band masks, and metadata in NPZ or Zarr format."
)
_DEBUG_ENV = "ALCHEMI_DEBUG"

app = typer.Typer(add_completion=False)
data_app = typer.Typer(add_completion=False)
align_app = typer.Typer(add_completion=False)
app.add_typer(data_app, name="data")
app.add_typer(align_app, name="align")
_LOG = get_logger(__name__)


def _debug_enabled() -> bool:
    return os.getenv(_DEBUG_ENV, "").lower() in {"1", "true", "yes", "on"}


def _echo_error(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)


def _log_cli_exception(exc: Exception, context: str) -> None:
    if _debug_enabled():
        _LOG.exception("%s", exc)
    else:
        _LOG.error("%s failed: %s", context, exc)


def handle_cli_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    """Provide consistent logging and user-friendly errors for CLI commands."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (typer.BadParameter, typer.Exit, KeyboardInterrupt):
            raise
        except (
            FileNotFoundError,
            OSError,
            yaml.YAMLError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            _log_cli_exception(exc, func.__name__)
            _echo_error(str(exc))
        except Exception as exc:  # pragma: no cover - handled by debug path
            if _debug_enabled():
                raise
            _LOG.exception("Unexpected error while running %s: %s", func.__name__, exc)
            _echo_error(f"Unexpected error. Re-run with {_DEBUG_ENV}=1 for a traceback.")

        raise typer.Exit(code=1)

    return wrapper


def _print_version() -> None:
    try:
        pkg_version = metadata.version("alchemi-hsi")
    except metadata.PackageNotFoundError:
        pkg_version = _read_local_version()

    typer.echo(f"alchemi-hsi version: {pkg_version}")
    typer.echo("Supported sensors:")
    for sensor in _SUPPORTED_SENSORS:
        typer.echo(f"  - {sensor}")
    typer.echo(f"Canonical cubes: {_CANONICAL_DESC}")


def _read_local_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"

    try:
        data = tomllib.loads(pyproject.read_text())
    except Exception:  # pragma: no cover - defensive fallback
        return "unknown"

    return str(data.get("project", {}).get("version", "unknown"))


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version information and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        _print_version()
        raise typer.Exit()


@app.command("version")  # type: ignore[misc]
def version_command() -> None:
    """Print version and supported data details."""

    _print_version()


@app.command()  # type: ignore[misc]
@handle_cli_exceptions
def validate_srf(root: str = "data/srf", sensor: str = "emit") -> None:
    reg = SRFRegistry(root)
    srf = reg.get(sensor)
    ints = srf.row_integrals()
    _LOG.info("SRF integrals (first 5): %s", ints[:5])


@app.command()  # type: ignore[misc]
@handle_cli_exceptions
def validate_data(config: str = "configs/data.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    validate_dataset(cfg)
    validate_srf_dir(cfg.get("data", {}).get("srf_root", "data/srf"))


@app.command()  # type: ignore[misc]
@handle_cli_exceptions
def pretrain_mae(
    config: str = "configs/train.mae.yaml",
    no_spatial_mask: bool = typer.Option(
        False, "--no-spatial-mask", help="Disable spatial masking for MAE baseline"
    ),
    no_posenc: bool = typer.Option(
        False, "--no-posenc", help="Disable wavelength positional encoding for MAE baseline"
    ),
) -> None:
    seed_everything(42)
    run_pretrain_mae(config, no_spatial_mask=no_spatial_mask, no_posenc=no_posenc)


@align_app.command("train")  # type: ignore[misc]
@handle_cli_exceptions
def align_train(
    cfg: str = typer.Option("configs/phase2/alignment.yaml", "--cfg", "-c"),
    max_steps: int | None = typer.Option(None, "--max-steps", "-m"),
) -> None:
    """Run the Phase-2 alignment trainer."""
    trainer = AlignmentTrainer.from_yaml(cfg)
    trainer.train(max_steps=max_steps)


@app.command()  # type: ignore[misc]
@handle_cli_exceptions
def evaluate(config: str = "configs/eval.yaml") -> None:
    run_eval(config)


@data_app.command("info")  # type: ignore[misc]
@handle_cli_exceptions
def data_info(path: Path) -> None:
    """Inspect a hyperspectral cube and print a short summary."""

    cube = _load_cube(path)
    _print_cube_summary(cube)


@data_app.command("to-canonical")  # type: ignore[misc]
@handle_cli_exceptions
def data_to_canonical(path: Path, out: str = typer.Option("npz", "--out")) -> None:
    """Write the canonical representation of a hyperspectral cube."""

    cube = _load_cube(path)
    fmt, destination = _resolve_output_path(path, out)

    if fmt == "npz":
        cube.to_npz(destination)
    elif fmt == "zarr":
        cube.to_zarr(destination)
    else:  # pragma: no cover - guard against misuse
        raise typer.BadParameter(f"Unsupported output format: {fmt}")

    typer.echo(f"Wrote canonical {fmt.upper()} cube to {destination}")


@dataclass
class _SniffResult:
    loader: Callable[[], xr.Dataset]
    description: str


def _load_cube(path: Path) -> Cube:
    path = path.expanduser()
    sniffed = _sniff_dataset(path)
    if sniffed is None:
        raise typer.BadParameter(f"Could not determine sensor for: {path}")
    dataset = sniffed.loader()
    return Cube.from_xarray(dataset)


def _resolve_output_path(source: Path, out: str) -> tuple[str, Path]:
    out_lower = out.lower()
    if out_lower in {"npz", "zarr"}:
        destination = source.with_name(f"{source.stem}.canonical.{out_lower}")
        return out_lower, destination

    destination = Path(out)
    suffix = destination.suffix.lower().lstrip(".")
    if suffix not in {"npz", "zarr"}:
        raise typer.BadParameter("Output must be a .npz or .zarr path or format keyword")
    return suffix, destination


def _sniff_dataset(path: Path) -> _SniffResult | None:
    for sniff in (
        _sniff_emit,
        _sniff_enmap,
        _sniff_avirisng,
        _sniff_hytes,
        _sniff_mako,
    ):
        result = sniff(path)
        if result is not None:
            return result
    return None


def _sniff_emit(path: Path) -> _SniffResult | None:
    if not path.is_file():
        return None
    name = path.name.lower()
    if "emit" not in name and path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    return _SniffResult(lambda: load_emit_l1b(str(path)), "EMIT L1B")


def _sniff_enmap(path: Path) -> _SniffResult | None:
    if path.is_dir():
        vnir = _find_first(path, "vnir")
        swir = _find_first(path, "swir")
        if vnir and swir:
            return _SniffResult(
                lambda: load_enmap_l1b(str(vnir), str(swir)),
                "EnMAP L1B",
            )
        return None

    name = path.name.lower()
    if "enmap" not in name:
        return None

    partner = _find_partner(path)
    if partner is None:
        return None

    if "vnir" in name:
        return _SniffResult(
            lambda: load_enmap_l1b(str(path), str(partner)),
            "EnMAP L1B",
        )
    return _SniffResult(
        lambda: load_enmap_l1b(str(partner), str(path)),
        "EnMAP L1B",
    )


def _sniff_avirisng(path: Path) -> _SniffResult | None:
    if not path.is_file():
        return None
    name = path.name.lower()
    if not ("aviris" in name or name.startswith("ang")):
        if path.suffix.lower() not in {".h5", ".he5", ".hdf", ".nc"}:
            return None

    return _SniffResult(lambda: load_avirisng_l1b(str(path)), "AVIRIS-NG L1B")


def _sniff_hytes(path: Path) -> _SniffResult | None:
    if not path.is_file():
        return None
    name = path.name.lower()
    if "hytes" not in name and path.suffix.lower() not in {".nc", ".cdf"}:
        return None

    return _SniffResult(lambda: load_hytes_l1b_bt(str(path)), "HyTES L1B")


def _sniff_mako(path: Path) -> _SniffResult | None:
    if not path.exists():
        return None

    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix not in {".hdr", ".img", ".dat"} and "mako" not in name and "comex" not in name:
        return None

    if "ace" in name:
        return _SniffResult(lambda: open_mako_ace(path), "Mako ACE")
    if "bt" in name or "btemp" in name:
        return _SniffResult(lambda: open_mako_btemp(path), "Mako BTEMP")
    return _SniffResult(lambda: open_mako_l2s(path), "Mako L2S")


def _find_first(root: Path, keyword: str) -> Path | None:
    keyword = keyword.lower()
    for candidate in sorted(root.iterdir()):
        if keyword in candidate.name.lower():
            return candidate
    return None


def _find_partner(path: Path) -> Path | None:
    keyword = "swir" if "vnir" in path.name.lower() else "vnir"
    return _find_first(path.parent, keyword)


def _print_cube_summary(cube: Cube) -> None:
    # Prefer explicit attributes, fall back to metadata if missing
    sensor = cube.sensor or cube.metadata.get("sensor", "unknown")
    units = cube.units or cube.metadata.get("units", "unknown")

    typer.echo(f"Sensor: {sensor}")
    typer.echo(f"Quantity: {cube.quantity} [{units}]")

    axis_parts = [f"{name}={size}" for name, size in zip(cube.axes, cube.shape, strict=True)]
    typer.echo("Shape: " + ", ".join(axis_parts))

    axis_nm = cube.wavelength_nm
    if axis_nm is not None and axis_nm.size:
        start = float(axis_nm[0])
        end = float(axis_nm[-1])
        typer.echo(f"Spectral range: {start:.2f}-{end:.2f} nm ({axis_nm.size} bands)")

    if cube.band_mask is not None:
        total = cube.band_mask.size
        good = int(np.count_nonzero(cube.band_mask))
        typer.echo(f"Band mask: {good}/{total} bands marked good")

    # Print metadata, excluding things already surfaced above
    metadata = dict(cube.metadata) if cube.metadata else {}
    metadata.pop("sensor", None)
    metadata.pop("units", None)
    if metadata:
        meta_json = json.dumps(_json_ready(metadata), sort_keys=True, default=str)
        typer.echo(f"Metadata: {meta_json}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


if __name__ == "__main__":
    app()
