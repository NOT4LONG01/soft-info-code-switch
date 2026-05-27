"""Backward-compat re-exports for the historical ``soft_info.decoder`` namespace."""

from .base import SoftOutputsDecoder
from .bplsd import SoftOutputsBpLsdDecoder
from .matching import SoftOutputsMatchingDecoder
from ..analysis.clusters import compute_cluster_stats

# Export all classes
__all__ = [
    "compute_cluster_stats",
    "SoftOutputsDecoder",
    "SoftOutputsBpLsdDecoder",
    "SoftOutputsMatchingDecoder",
]
