"""Prediction heads operating on pooled embeddings."""

from .banddepth import BandDefinition, BandDepthHead, load_banddepth_config

__all__ = ["BandDefinition", "BandDepthHead", "load_banddepth_config"]
