"""Utilities for working with EMIT L2B mineral (MIN) products."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

__all__ = [
    "EMIT_TO_USGS",
    "iter_high_confident_pixels",
    "load_emit_l2b",
    "map_emit_group_to_splib",
]

# Mapping between EMIT mineral groups and candidate USGS SPLIB entries.  The list
# is intentionally small - it can be extended as crosswalks between the two
# catalogs become available.
EMIT_TO_USGS: dict[str, list[str]] = {
    "ILLITE_MUSCOVITE_GROUP": ["Illite GDS82", "Muscovite HS315.3B"],
    "KAOLINITE_GROUP": ["Kaolinite KGa-1", "Kaolinite KGa-2"],
    "SMECTITE_GROUP": ["Montmorillonite SWy-1"],
    "ALUNITE_GROUP": ["Alunite GDS82"],
}


def _drop_empty_variables(ds: xr.Dataset) -> xr.Dataset:
    """Drop variables that contain no valid (non-NaN) samples."""
    empty_vars: list[str] = []
    for name, data in ds.data_vars.items():
        # ``count`` ignores NaN values and masked values.  When the count is zero
        # we know there were no valid pixels in the cube for that variable.
        if int(data.count()) == 0:
            empty_vars.append(name)
    if empty_vars:
        ds = ds.drop_vars(empty_vars)
    return ds


def _ensure_expected_fields(ds: xr.Dataset) -> None:
    """Validate that the dataset looks like an EMIT L2B mineral product."""
    if not {"y", "x"}.issubset(ds.dims):
        raise ValueError(
            "Expected spatial dimensions ('y', 'x') in EMIT L2B dataset, " f"found {tuple(ds.dims)}"
        )

    mineral_vars = [name for name in ds.data_vars if "mineral" in name.lower()]
    if not mineral_vars:
        raise ValueError(
            "Dataset does not contain any mineral identification variables. "
            "Expected at least one variable containing 'mineral' in its name."
        )

    band_depth_vars = [name for name in ds.data_vars if "band_depth" in name.lower()]
    if not band_depth_vars:
        raise ValueError(
            "Dataset does not contain any band-depth variables. "
            "Expected at least one variable containing 'band_depth' in its name."
        )

    quality_vars = [
        name
        for name in ds.data_vars
        if any(key in name.lower() for key in ("r2", "score", "quality"))
    ]
    if not quality_vars:
        raise ValueError("Dataset does not contain a quality/fit metric (e.g. 'fit_r2').")


def load_emit_l2b(path: str | Path, *, drop_empty: bool = True) -> xr.Dataset:
    """Load an EMIT L2B mineral (MIN) product into an :class:`xarray.Dataset`.

    Parameters
    ----------
    path
        Path to the EMIT L2B NetCDF file on disk.
    drop_empty
        When ``True`` (default), variables that contain no valid pixels (all NaN
        or masked) are removed from the returned dataset.  Keeping only
        informative variables simplifies downstream weak-supervision workflows.

    Returns
    -------
    ds
        Dataset containing spatial dimensions ``('y', 'x')`` alongside mineral
        identification, band-depth diagnostics, and fit quality metrics.

    Notes
    -----
    * The loader leaves metadata untouched to preserve provenance information.
    * No attempt is made to align the dataset to L1B grids; consumers can handle
      co-registration separately.
    * If the dataset does not resemble an EMIT L2B MIN product a descriptive
      ``ValueError`` is raised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        ds_loaded = ds.load()

    _ensure_expected_fields(ds_loaded)

    if drop_empty:
        ds_loaded = _drop_empty_variables(ds_loaded)

    return ds_loaded


def map_emit_group_to_splib(group_name: str) -> list[str]:
    """Return candidate USGS SPLIB names for a given EMIT mineral group."""
    return EMIT_TO_USGS.get(group_name, []).copy()


def _as_numpy(data: xr.DataArray) -> NDArray[np.generic]:
    """Convert an xarray ``DataArray`` to a plain :class:`numpy.ndarray`."""
    values = data.load().to_numpy()
    if isinstance(values, np.ma.MaskedArray):
        values = values.filled(np.nan)
    return np.asarray(values)


def _is_missing(value: object) -> bool:
    """Heuristic check for missing entries in the mineral group field."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value.strip() == ""
    return False


def iter_high_confident_pixels(
    ds_l2b: xr.Dataset,
    *,
    r2_min: float = 0.9,
    mineral_var: str = "mineral_group",
    r2_var: str = "fit_r2",
) -> Iterator[tuple[int, int, str, float]]:
    """Yield spatial indices and weak mineral labels above the confidence cut.

    Parameters
    ----------
    ds_l2b
        EMIT L2B mineral dataset produced by :func:`load_emit_l2b`.
    r2_min
        Minimum acceptable fit quality (e.g. coefficient of determination).
    mineral_var
        Name of the variable containing the mineral group identifiers.
    r2_var
        Name of the variable containing the fit quality scores.

    Yields
    ------
    tuple[int, int, str, float]
        ``(y_idx, x_idx, mineral_group, r2)`` for pixels that satisfy the
        ``r2_min`` constraint.  These are *weak labels* - they indicate likely
        mineral dominance but are not guaranteed to be pure endmembers.
    """
    if mineral_var not in ds_l2b:
        raise ValueError(f"Dataset missing mineral variable '{mineral_var}'.")
    if r2_var not in ds_l2b:
        raise ValueError(f"Dataset missing quality variable '{r2_var}'.")

    minerals = _as_numpy(ds_l2b[mineral_var])
    r2_scores = _as_numpy(ds_l2b[r2_var])

    if minerals.shape != r2_scores.shape:
        raise ValueError(
            "Mineral and quality arrays must share the same shape. "
            f"Got {minerals.shape!r} and {r2_scores.shape!r}."
        )

    if minerals.ndim != 2:
        raise ValueError(
            "Expected a 2-D mineral grid indexed by ('y', 'x'); "
            f"received an array with shape {minerals.shape!r}."
        )

    for (y_idx, x_idx), mineral_value in np.ndenumerate(minerals):
        r2_value = float(r2_scores[y_idx, x_idx])
        if np.isnan(r2_value) or r2_value < r2_min:
            continue

        if _is_missing(mineral_value):
            continue

        if isinstance(mineral_value, (bytes, bytearray)):
            mineral_str = mineral_value.decode("utf-8", errors="ignore")
        else:
            mineral_str = str(mineral_value)

        yield int(y_idx), int(x_idx), mineral_str, r2_value
