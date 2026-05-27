"""
soft_info.analysis
------------------
Post-processing utilities consumed by decoders and pipelines: cluster labelling,
sliding-window norm calculators, logical-error-distribution sampling, and
plotting.

Submodules
~~~~~~~~~~
- ``clusters``       : Cluster labelling (scipy + igraph) and
                       :func:`compute_cluster_stats`.
- ``sliding_window`` : Cached committed-cluster norm calculator for sliding-window
                       decoding.
- ``distribution``   : :func:`collect_logical_error_distribution` and helpers
                       (:func:`logical_class_to_index`, :func:`normalize_distribution`, …).
- ``plotting``       : Standalone matplotlib/seaborn helpers (legacy `tools.py`).
"""

from .clusters import compute_cluster_stats, label_clusters, label_clusters_igraph
from .distribution import (
    collect_logical_error_distribution,
    collect_logical_error_distribution_fast,
    index_to_logical_class,
    logical_class_to_index,
    normalize_distribution,
)

__all__ = [
    "compute_cluster_stats",
    "label_clusters",
    "label_clusters_igraph",
    "collect_logical_error_distribution",
    "collect_logical_error_distribution_fast",
    "index_to_logical_class",
    "logical_class_to_index",
    "normalize_distribution",
]
