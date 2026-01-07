"""
LDPC Post-Selection Package.

A package for decoding quantum LDPC codes with cluster-based post-selection,
implementing confidence metrics for quantum error correction decoding.
"""

from ldpc_post_selection.bplsd_decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.logical_error_distribution import (
    collect_logical_error_distribution,
    collect_logical_error_distribution_fast,
    index_to_logical_class,
    logical_class_to_index,
    normalize_distribution,
)
from ldpc_post_selection.matching_decoder import SoftOutputsMatchingDecoder

__all__ = [
    "SoftOutputsBpLsdDecoder",
    "SoftOutputsMatchingDecoder",
    "collect_logical_error_distribution",
    "collect_logical_error_distribution_fast",
    "logical_class_to_index",
    "index_to_logical_class",
    "normalize_distribution",
]
