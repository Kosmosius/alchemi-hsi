from .aux import AuxHead, AuxOutputs
from .domain import DomainDiscriminator
from .gas import GasHead, GasOutput
from .gas_head import GasHead as LegacyGasHead
from .id_head import IDHead
from .solids import SolidsHead, SolidsOutput
from .unmix_head import LinearUnmixHead

__all__ = [
    "AuxHead",
    "AuxOutputs",
    "DomainDiscriminator",
    "GasHead",
    "GasOutput",
    "IDHead",
    "LinearUnmixHead",
    "LegacyGasHead",
    "SolidsHead",
    "SolidsOutput",
]
