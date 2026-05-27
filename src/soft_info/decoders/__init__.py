"""
soft_info.decoders
------------------
Decoder primitives and Sinter-compatible wrappers.

Submodules
~~~~~~~~~~
- ``base``     : :class:`SoftOutputsDecoder` base class (matrix / circuit construction).
- ``bplsd``    : :class:`SoftOutputsBpLsdDecoder` (BP+LSD with cluster stats / gap proxies).
- ``matching`` : :class:`SoftOutputsMatchingDecoder` (PyMatching with exhaustive 2^k gap).
- ``sinter``   : Sinter ``CompiledDecoder`` wrappers for mwpf, tesseract, bp_osd, relay_bp,
                 plus :func:`build_decoder` and the ``read_trace`` helper.
- ``legacy``   : Back-compat re-exports of the historical ``soft_info.decoder`` names.
"""

from .base import SoftOutputsDecoder
from .bplsd import SoftOutputsBpLsdDecoder
from .matching import SoftOutputsMatchingDecoder

__all__ = [
    "SoftOutputsDecoder",
    "SoftOutputsBpLsdDecoder",
    "SoftOutputsMatchingDecoder",
]
