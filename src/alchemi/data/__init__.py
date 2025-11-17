from importlib import import_module

from . import io
from .cube import Cube, GeoInfo, geo_from_attrs
from .datasets import PairingDataset, SpectrumDataset
from .gas_sim import inject_synthetic_plume
from .transforms import RandomBandDropout, SpectralNoise
from .validators import validate_dataset, validate_srf_dir

__all__ = [
    "LabSensorCache",
    "PairBuilder",
    "PairingDataset",
    "RandomBandDropout",
    "SpectralNoise",
    "SpectrumDataset",
    "Cube",
    "GeoInfo",
    "inject_synthetic_plume",
    "io",
    "geo_from_attrs",
    "validate_dataset",
    "validate_srf_dir",
]

_LAZY_ATTRS = {
    "LabSensorCache": ("alchemi.data.pairing", "LabSensorCache"),
    "PairBuilder": ("alchemi.data.pairing", "PairBuilder"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'alchemi.data' has no attribute {name!r}")
    module = import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_ATTRS))
