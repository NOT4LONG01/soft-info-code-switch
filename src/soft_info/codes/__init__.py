"""
soft_info.codes
---------------
Quantum code definitions, Stim circuit construction, noise models, and
DEM-to-matrix tools.

Submodules
~~~~~~~~~~
- ``registry``      : :class:`CodeRegistry`, :class:`CSSCode`, alist loader, and
                      :func:`schedule_dir` for locating optimised CNOT schedules.
- ``circuit``       : :func:`generate_experiment_with_noise` + schedule loaders.
- ``noise``         : Noise-model variants (phenomenological, depolarising, SI1000,
                      Bravyi).
- ``stim_tools``    : :func:`dem_to_parity_check` (DEM → sparse H + obs matrix + priors)
                      and related Stim helpers.
- ``leaf_factored`` : Observable-gauge rewrite stripping leaf-qubit support.
"""

from .registry import (
    CodeRegistry,
    CSSCode,
    schedule_dir,
    readAlist,
    TRIANGULAR_DIR,
    TETRAHEDRAL_DIR,
    TRIANGULAR_DICT,
    TETRAHEDRAL_DICT,
)

__all__ = [
    "CodeRegistry",
    "CSSCode",
    "schedule_dir",
    "readAlist",
    "TRIANGULAR_DIR",
    "TETRAHEDRAL_DIR",
    "TRIANGULAR_DICT",
    "TETRAHEDRAL_DICT",
]
