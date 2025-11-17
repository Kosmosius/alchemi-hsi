from __future__ import annotations

from pathlib import Path

import numpy as np

from alchemi.data.io import load_splib
from alchemi.types import SpectrumKind


def _write(tmp_path: Path, relative: str, content: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_splib_units_and_bounds(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "forsterite.txt",
        """
        # name: Forsterite
        # alias: Fo90 Olivine
        # wavelength_unit: micron
        # reflectance_unit: percent
        wavelength,reflectance
        0.35,10
        0.40,50
        0.45,100
        """,
    )

    catalog = load_splib(tmp_path)

    assert "Forsterite" in catalog
    spectra = catalog["Forsterite"]
    assert len(spectra) == 1

    spectrum = spectra[0]
    np.testing.assert_allclose(spectrum.wavelengths.nm, np.array([350.0, 400.0, 450.0]))
    np.testing.assert_allclose(spectrum.values, np.array([0.10, 0.50, 1.00]))
    assert spectrum.kind is SpectrumKind.REFLECTANCE
    assert spectrum.meta["sensor"] == "LAB"
    assert spectrum.meta["quantity"] == "reflectance"
    assert set(spectrum.meta["aliases"]) >= {"Forsterite", "Fo90 Olivine", "forsterite"}

    cache_path = tmp_path / ".splib-cache.npz"
    assert cache_path.exists()

    cached = load_splib(tmp_path)
    np.testing.assert_allclose(
        cached["Forsterite"][0].wavelengths.nm, np.array([350.0, 400.0, 450.0])
    )


def test_alias_resolution(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "hematite_sample.csv",
        """
        # name: Hematite
        # alias: Fe2O3; Red Ochre
        wavelength,reflectance
        350,0.2
        360,0.3
        370,0.4
        """,
    )

    _write(
        tmp_path,
        "variants/hematite_variant.txt",
        """
        # canonical_name: Hematite
        # alias: Specular Haematite
        # wavelength_unit: micron
        wavelength reflectance
        0.38 0.25
        0.39 0.27
        0.40 0.32
        """,
    )

    catalog = load_splib(tmp_path)

    assert catalog.canonical_name("Fe2O3") == "Hematite"
    assert catalog.canonical_name("specular haematite") == "Hematite"
    assert len(catalog["Hematite"]) == 2

    resolved = catalog.resolve("Fe2O3")
    assert len(resolved) == 2
    assert all(spec.kind is SpectrumKind.REFLECTANCE for spec in resolved)

    aliases = set(catalog.aliases["Hematite"])
    assert {"Hematite", "Fe2O3", "Red Ochre", "Specular Haematite"}.issubset(aliases)
