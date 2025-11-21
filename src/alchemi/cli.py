from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import typer
import xarray as xr
import yaml

from .data.cube import Cube
from .data.io import load_avirisng_l1b, load_emit_l1b, load_enmap_l1b, load_hytes_l1b_bt
from .data.validators import validate_dataset, validate_srf_dir
from .io.mako import open_mako_ace, open_mako_btemp, open_mako_l2s
from .srf import SRFRegistry
from .train.alignment_trainer import AlignmentTrainer
from .training.trainer import run_eval, run_pretrain_mae
from .utils.logging import get_logger

app = typer.Typer(add_completion=False)
data_app = typer.Typer(add_completion=False)
align_app = typer.Typer(add_completion=False)
app.add_typer(data_app, name="data")
app.add_typer(align_app, name="align")
_LOG = get_logger(__name__)


@app.command()  # type: ignore[misc]
def validate_srf(root: str = "data/srf", sensor: str = "emit") -> None:
    reg = SRFRegistry(root)
    srf = reg.get(sensor)
    ints = srf.row_integrals()
    _LOG.info("SRF integrals (first 5): %s", ints[:5])


@app.command()  # type: ignore[misc]
def validate_data(config: str = "configs/data.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    validate_dataset(cfg)
    validate_srf_dir(cfg.get("data", {}).get("srf_root", "data/srf"))


@app.command(help="Synthetic MAE sandbox for masking/throughput baselines.")  # type: ignore[misc]
def pretrain_mae(
    config: str = "configs/train.mae.yaml",
    no_spatial_mask: bool = typer.Option(
        False, "--no-spatial-mask", help="Disable spatial masking for MAE baseline"
    ),
    no_posenc: bool = typer.Option(
        False, "--no-posenc", help="Disable wavelength positional encoding for MAE baseline"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Override random seed configured in the YAML file"
    ),
) -> None:
    run_pretrain_mae(
        config,
        no_spatial_mask=no_spatial_mask,
        no_posenc=no_posenc,
        seed_override=seed,
    )


@align_app.command("train")  # type: ignore[misc]
def align_train(
    cfg: str = typer.Option("configs/phase2/alignment.yaml", "--cfg", "-c"),
    max_steps: int | None = typer.Option(None, "--max-steps", "-m"),
    seed: int | None = typer.Option(
        None, "--seed", help="Override random seed configured in the YAML file"
    ),
) -> None:
    """Run the mainline CLIP-style alignment trainer used for the encoder."""
    trainer = AlignmentTrainer.from_yaml(cfg, seed_override=seed)
    trainer.train(max_steps=max_steps)


@app.command()  # type: ignore[misc]
def evaluate(config: str = "configs/eval.yaml") -> None:
    run_eval(config)


SensorLiteral = Literal["emit", "enmap", "avirisng", "hytes", "mako"]
_SENSOR_CHOICES: tuple[SensorLiteral, ...] = (
    "emit",
    "enmap",
    "avirisng",
    "hytes",
    "mako",
)
SensorChoice = typer.Option(  # type: ignore[misc]
    None,
    "--sensor",
    case_sensitive=False,
    help="Explicitly specify the sensor instead of sniffing from the path.",
    show_default=False,
)


@data_app.command("info")  # type: ignore[misc]
def data_info(path: Path, sensor: SensorLiteral | None = SensorChoice) -> None:
    """Inspect a hyperspectral cube and print a short summary."""

    cube = _load_cube(path, sensor)
    _print_cube_summary(cube)


@data_app.command("to-canonical")  # type: ignore[misc]
def data_to_canonical(
    path: Path,
    out: str = typer.Option("npz", "--out"),
    sensor: SensorLiteral | None = SensorChoice,
) -> None:
    """Write the canonical representation of a hyperspectral cube."""

    cube = _load_cube(path, sensor)
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


def _load_cube(path: Path, sensor: SensorLiteral | None = None) -> Cube:
    path = path.expanduser()
    sniffed = _resolve_loader(path, sensor)
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


def _resolve_loader(path: Path, sensor: SensorLiteral | None) -> _SniffResult:
    if sensor is not None:
        return _loader_for_sensor(path, sensor)

    sniffed = _sniff_dataset(path)
    if sniffed is None:
        valid = ", ".join(_SENSOR_CHOICES)
        raise typer.BadParameter(
            f"Could not determine sensor for: {path}. "
            f"Specify --sensor to override. Valid sensors: {valid}"
        )
    return sniffed


def _loader_for_sensor(path: Path, sensor: SensorLiteral) -> _SniffResult:
    sensor_lower = sensor.lower()
    if sensor_lower == "emit":
        return _SniffResult(lambda: load_emit_l1b(str(path)), "EMIT L1B")
    if sensor_lower == "avirisng":
        return _SniffResult(lambda: load_avirisng_l1b(str(path)), "AVIRIS-NG L1B")
    if sensor_lower == "hytes":
        return _SniffResult(lambda: load_hytes_l1b_bt(str(path)), "HyTES L1B")
    if sensor_lower == "enmap":
        result = _sniff_enmap(path)
    elif sensor_lower == "mako":
        result = _sniff_mako(path)
    else:  # pragma: no cover - guarded by typer Choice
        valid = ", ".join(_SENSOR_CHOICES)
        raise typer.BadParameter(
            f"Unknown sensor '{sensor}'. Valid sensors: {valid}"
        )

    if result is None:
        raise typer.BadParameter(f"Could not load sensor '{sensor}' for: {path}")
    return result


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
