"""
soft_info.codes.registry
------------------------
OOP registry for the QEC code families exposed by this project (currently
``triangular`` and ``tetrahedral``; other families remain defined but are not
registered until their alist data lands in ``src/pcms/``).

Dataclasses
-----------
CSSCode
    CSS code container; fields: code_type, n, k, d, variant, Hx, Hz.

Registry
--------
CodeRegistry
    load(code_type, n, variant='base') -> CSSCode   Load and cache a code instance.
    choices()                          -> list[str]  All registered code_type strings.
    n_values(code_type)                -> list[int]  Valid n values for a code type.
    available(code_type)               -> list[tuple[int, str]]  All (n, variant) pairs.
    iter_all(code_type)                -> generator[CSSCode]     Iterate every instance.
    register(family_cls)                             Register a new code family class.

Code families
-------------
GO03Code              — GO03 self-dual bicycle codes; Hx == Hz via GF(2) null-space.
CappedColorCode       — Capped color codes loaded from alist files.
EQRIsoDualCode        — Extended QR iso-dual codes; separate Hx/Hz alist files.
EQRSelfDualCSSCode    — Extended QR self-dual CSS codes; separate Hx/Hz alist files.
JA25Code              — JA25 transversal-T codes; variants discovered from filesystem.
TetrahedralCode       — 3D tetrahedral color codes loaded from alist files.
TriangularCode        — 2D triangular color codes loaded from alist files.
QuantumSelfDualCSSCode — [[n,1,d]] self-dual CSS codes; multiple d per n, opt_H variant.

Distance dicts (exported)
-------------------------
GO03_DICT, EQR_ISO_DICT, EQR_SD_CSS_DICT, JA25_DICT, CAPPED_COLOR_DICT,
TETRAHEDRAL_DICT, TRIANGULAR_DICT
    Map n -> theoretical code distance d for each family.

Path constants (exported)
-------------------------
GO03_DIR, EQR_ISO_DIR, EQR_SD_CSS_DIR, JA25_DIR, CAPPED_COLOR_DIR,
TETRAHEDRAL_DIR, TRIANGULAR_DIR, QSD_DIR, _OBSOLETTE_DIR
    Absolute paths to alist file directories.

Internal utilities
------------------
readAlist(path)              -> np.ndarray   Parse an alist file into a binary matrix.
check_commutativity(Hx, Hz)  -> bool         Verify Hx @ Hz.T == 0 over GF(2).
_gf2_rank(H)                 -> int          Pure-numpy GF(2) row reduction.
_compute_k(Hx, Hz)           -> int          Logical qubit count k = n - rank(Hx) - rank(Hz).
"""

import os
import glob
import numpy as np
import galois
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR       = os.path.dirname(_THIS_DIR)
_SRC_DIR       = os.path.dirname(_PKG_DIR)
_PCMS_DIR      = os.path.join(_SRC_DIR, 'pcms')
_CODES_LIB_DIR = os.path.join(os.path.dirname(_SRC_DIR), 'codes_lib')

_OBSOLETTE_DIR    = os.path.join(_CODES_LIB_DIR, "obsolette")
GO03_DIR          = os.path.join(_OBSOLETTE_DIR,  "GO03_self_dual")
EQR_ISO_DIR       = os.path.join(_OBSOLETTE_DIR,  "eQR_iso_dual_CSS")
EQR_SD_CSS_DIR    = os.path.join(_OBSOLETTE_DIR,  "eQR_self_dual_CSS")
JA25_DIR          = os.path.join(_CODES_LIB_DIR,  "JA25_transversal_T")
CAPPED_COLOR_DIR  = os.path.join(_CODES_LIB_DIR,  "capped_color_codes")
TETRAHEDRAL_DIR   = os.path.join(_PCMS_DIR,       "tetrahedral_codes")
TRIANGULAR_DIR    = os.path.join(_PCMS_DIR,       "triangular_codes")
QSD_DIR           = os.path.join(_CODES_LIB_DIR,  "quantum_self_dual_CSS")


def schedule_dir(code_type: str) -> str:
    """Directory holding measurement schedules for `code_type`, colocated with the alist matrices in pcms/."""
    return os.path.join(_PCMS_DIR, f"{code_type}_codes")

GO03_DICT         = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}
JA25_DICT         = {15:3, 49:5, 95:7, 185:9, 189:9, 279:11, 283:11, 441:13, 599:15}
CAPPED_COLOR_DICT = {49:5, 53:5}
EQR_ISO_DICT      = {17:3, 41:7, 73:11}
EQR_SD_CSS_DICT   = {7:3, 23:7, 47:11}
TETRAHEDRAL_DICT  = {15:3, 65:5, 175:7, 369:9}
TRIANGULAR_DICT   = {7:3, 19:5, 37:7, 61:9, 91:11}



def readAlist(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        lines = [line.split() for line in f if line.strip() and not line.strip().startswith('#')]
    if not lines:
        raise ValueError(f"Alist file at {path} is empty.")
    n, m = int(lines[0][0]), int(lines[0][1])
    matrix = np.zeros((m, n), dtype=np.uint8)
    for col_idx in range(n):
        for row_val in lines[4 + col_idx]:
            row_idx = int(row_val)
            if row_idx > 0:
                matrix[row_idx - 1, col_idx] = 1
    return matrix



def check_commutativity(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    if Hx.shape[1] != Hz.shape[1]:
        return False
    return not np.any(np.mod(Hx.astype(np.int32) @ Hz.T.astype(np.int32), 2))


def _gf2_rank(H: np.ndarray) -> int:
    M = H.astype(bool).copy()
    rank = 0
    for col in range(M.shape[1]):
        pivot_rows = np.where(M[rank:, col])[0]
        if not pivot_rows.size:
            continue
        p = pivot_rows[0] + rank
        M[[rank, p]] = M[[p, rank]]
        elim = np.where(M[:, col])[0]
        elim = elim[elim != rank]
        M[elim] ^= M[rank]
        rank += 1
        if rank == M.shape[0]:
            break
    return rank


def _compute_k(Hx: np.ndarray, Hz: np.ndarray) -> int:
    n = Hx.shape[1]
    if np.array_equal(Hx, Hz):
        return n - 2 * _gf2_rank(Hx)
    return n - _gf2_rank(Hx) - _gf2_rank(Hz)



@dataclass
class CSSCode:
    code_type: str
    n: int
    k: int
    d: int
    Hx: np.ndarray
    Hz: np.ndarray
    variant: str = 'base'

    def __repr__(self):
        return f"[[{self.n}, {self.k}, {self.d}]] {self.code_type} [{self.variant}]"



class _LocalCodeFamily:
    code_type: ClassVar[str] = ''
    _dir: ClassVar[str] = ''
    _dist_dict: ClassVar[dict] = {}
    _if_self_dual: ClassVar[bool] = False
    _alist_suffix: ClassVar[str] = '.alist'

    @classmethod
    def distance(cls, n: int) -> int:
        return cls._dist_dict[n]

    @classmethod
    def load(cls, n: int, variant: str = 'base') -> 'CSSCode':
        d = cls._dist_dict.get(n, 0)
        suffix = cls._alist_suffix

        if cls._if_self_dual:
            path = os.path.join(cls._dir, f"n{n}_d{d}{suffix}")
            try:
                F2 = galois.GF(2)
                GenMat = F2(readAlist(path))
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing self-dual file: {path}")
            G_punctured = GenMat[:, :-1]
            Hz = Hx = np.array(G_punctured.null_space(), dtype=np.uint8)
        else:
            variant_suffix = f"_{variant}" if variant and variant != 'base' else ''
            path_x = os.path.join(cls._dir, f"n{n}_d{d}_Hx{variant_suffix}{suffix}")
            path_z = os.path.join(cls._dir, f"n{n}_d{d}_Hz{variant_suffix}{suffix}")
            try:
                Hx = readAlist(path_x)
                Hz = readAlist(path_z)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing matrices in {cls._dir} for n={n}, d={d}, variant={variant!r}"
                )

        if not check_commutativity(Hx, Hz):
            print(f"Warning: CSS commutativity check failed for {cls.code_type} n={n}!")

        k = _compute_k(Hx, Hz)
        return CSSCode(
            code_type=cls.code_type,
            n=int(Hx.shape[1]),
            k=k, d=d,
            Hx=np.array(Hx, dtype=np.uint8),
            Hz=np.array(Hz, dtype=np.uint8),
            variant=variant,
        )

    @classmethod
    def available(cls) -> List[Tuple[int, str]]:
        return [(n, 'base') for n in cls._dist_dict]

    @classmethod
    def n_values(cls) -> List[int]:
        return list(cls._dist_dict.keys())



class GO03Code(_LocalCodeFamily):
    code_type = 'go03_self_dual'
    _dir = GO03_DIR
    _dist_dict = GO03_DICT
    _if_self_dual = True


class CappedColorCode(_LocalCodeFamily):
    code_type = 'capped_color_code'
    _dir = CAPPED_COLOR_DIR
    _dist_dict = CAPPED_COLOR_DICT
    _if_self_dual = False


class EQRIsoDualCode(_LocalCodeFamily):
    code_type = 'eqr_iso_dual'
    _dir = EQR_ISO_DIR
    _dist_dict = EQR_ISO_DICT
    _if_self_dual = False


class EQRSelfDualCSSCode(_LocalCodeFamily):
    code_type = 'eqr_self_dual_css'
    _dir = EQR_SD_CSS_DIR
    _dist_dict = EQR_SD_CSS_DICT
    _if_self_dual = False


class JA25Code(_LocalCodeFamily):
    code_type = 'ja25_transversal_t'
    _dir = JA25_DIR
    _dist_dict = JA25_DICT
    _if_self_dual = False

    @classmethod
    def variants(cls, n: int) -> List[str]:
        d = cls._dist_dict.get(n, 0)
        suffix = cls._alist_suffix
        variants = []
        if os.path.exists(os.path.join(cls._dir, f"n{n}_d{d}_Hx{suffix}")):
            variants.append('base')
        for f in sorted(glob.glob(os.path.join(cls._dir, f"n{n}_d{d}_Hx_*{suffix}"))):
            variant_name = os.path.basename(f)[len(f"n{n}_d{d}_Hx_"):-len(suffix)]
            variants.append(variant_name)
        return variants

    @classmethod
    def load(cls, n: int, variant: str = 'base') -> 'CSSCode':
        d = cls._dist_dict.get(n, 0)
        suffix = cls._alist_suffix
        variant_suffix = f"_{variant}" if variant and variant != 'base' else ''

        def _resolve(matrix: str) -> str:
            p = os.path.join(cls._dir, f"n{n}_d{d}_{matrix}{variant_suffix}{suffix}")
            if not os.path.exists(p) and variant.endswith('_opt'):
                base = variant[:-4]  # strip '_opt'
                fb_suffix = f"_{base}" if base and base != 'base' else ''
                p = os.path.join(cls._dir, f"n{n}_d{d}_{matrix}{fb_suffix}{suffix}")
            return p

        path_x = _resolve("Hx")
        path_z = _resolve("Hz")
        # Hz filename may use a different d for some variants (e.g. n49 tetra/opt).
        if not os.path.exists(path_z):
            candidates = sorted(glob.glob(os.path.join(cls._dir, f"n{n}_d*_Hz{variant_suffix}{suffix}")))
            if candidates:
                path_z = candidates[0]
        try:
            Hx = readAlist(path_x)
            Hz = readAlist(path_z)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing matrices in {cls._dir} for n={n}, d={d}, variant={variant!r}"
            )
        if not check_commutativity(Hx, Hz):
            print(f"Warning: CSS commutativity check failed for {cls.code_type} n={n}!")
        k = _compute_k(Hx, Hz)
        return CSSCode(
            code_type=cls.code_type,
            n=int(Hx.shape[1]),
            k=k, d=d,
            Hx=np.array(Hx, dtype=np.uint8),
            Hz=np.array(Hz, dtype=np.uint8),
            variant=variant,
        )

    @classmethod
    def available(cls) -> List[Tuple[int, str]]:
        return [(n, v) for n in cls._dist_dict for v in cls.variants(n)]



class TetrahedralCode(_LocalCodeFamily):
    code_type = 'tetrahedral'
    _dir = TETRAHEDRAL_DIR
    _dist_dict = TETRAHEDRAL_DICT
    _if_self_dual = False

    @classmethod
    def variants(cls, n: int) -> List[str]:
        d = cls._dist_dict.get(n, 0)
        suffix = cls._alist_suffix
        variants = []
        if os.path.exists(os.path.join(cls._dir, f"n{n}_d{d}_Hx{suffix}")):
            variants.append('base')
        for f in sorted(glob.glob(os.path.join(cls._dir, f"n{n}_d{d}_Hx_*{suffix}"))):
            variant_name = os.path.basename(f)[len(f"n{n}_d{d}_Hx_"):-len(suffix)]
            variants.append(variant_name)
        return variants

    @classmethod
    def available(cls) -> List[Tuple[int, str]]:
        return [(n, v) for n in cls._dist_dict for v in cls.variants(n)]


class TriangularCode(_LocalCodeFamily):
    code_type = 'triangular'
    _dir = TRIANGULAR_DIR
    _dist_dict = TRIANGULAR_DICT
    _if_self_dual = False


import re as _re

class QuantumSelfDualCSSCode:
    code_type = 'quantum_self_dual_css'

    @classmethod
    def _scan(cls) -> List[Tuple[int, int]]:
        result = []
        for f in sorted(glob.glob(os.path.join(QSD_DIR, "n*_k1_d*_Hx.alist"))):
            m = _re.match(r'n(\d+)_k1_d(\d+)_Hx\.alist', os.path.basename(f))
            if m:
                result.append((int(m.group(1)), int(m.group(2))))
        return result

    @classmethod
    def load(cls, n: int, variant: str = 'base') -> 'CSSCode':
        all_nd = cls._scan()
        ds_for_n = [d for (n_, d) in all_nd if n_ == n]
        if not ds_for_n:
            raise FileNotFoundError(f"No quantum_self_dual_css codes found for n={n}")

        is_opt = variant.endswith('_opt')
        v = variant[:-4] if is_opt else variant

        if v == 'base':
            if len(ds_for_n) != 1:
                raise ValueError(
                    f"n={n} has multiple codes (d={sorted(ds_for_n)}); "
                    f"specify variant='d{{d}}' or 'd{{d}}_opt'"
                )
            d = ds_for_n[0]
        elif v.startswith('d') and v[1:].isdigit():
            d = int(v[1:])
            if d not in ds_for_n:
                raise FileNotFoundError(
                    f"No quantum_self_dual_css code for n={n}, d={d}. "
                    f"Available: {sorted(ds_for_n)}"
                )
        else:
            raise ValueError(f"Unknown variant {variant!r} for quantum_self_dual_css n={n}")

        if is_opt:
            path = os.path.join(QSD_DIR, 'opt_H', f'n{n}_k1_d{d}_H_opt.alist')
            try:
                H = readAlist(path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing opt file: {path}")
            Hx = Hz = np.array(H, dtype=np.uint8)
            variant_out = f'd{d}_opt'
        else:
            path_x = os.path.join(QSD_DIR, f'n{n}_k1_d{d}_Hx.alist')
            path_z = os.path.join(QSD_DIR, f'n{n}_k1_d{d}_Hz.alist')
            try:
                Hx = readAlist(path_x)
                Hz = readAlist(path_z)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing quantum_self_dual_css files for n={n}, d={d}"
                )
            variant_out = 'base' if (len(ds_for_n) == 1 and variant == 'base') else f'd{d}'

        if not check_commutativity(Hx, Hz):
            print(f"Warning: commutativity check failed for quantum_self_dual_css n={n}!")
        k = _compute_k(Hx, Hz)
        return CSSCode(
            code_type=cls.code_type,
            n=int(Hx.shape[1]),
            k=k, d=d,
            Hx=np.array(Hx, dtype=np.uint8),
            Hz=np.array(Hz, dtype=np.uint8),
            variant=variant_out,
        )

    @classmethod
    def available(cls) -> List[Tuple[int, str]]:
        result = []
        for n, d in cls._scan():
            result.append((n, f'd{d}'))
            opt_path = os.path.join(QSD_DIR, 'opt_H', f'n{n}_k1_d{d}_H_opt.alist')
            if os.path.exists(opt_path):
                result.append((n, f'd{d}_opt'))
        return result

    @classmethod
    def n_values(cls) -> List[int]:
        return sorted({n for n, _ in cls._scan()})


class CodeRegistry:
    _families: ClassVar[dict] = {}
    _cache:    ClassVar[dict] = {}

    @classmethod
    def register(cls, family_cls):
        cls._families[family_cls.code_type] = family_cls

    @classmethod
    def load(cls, code_type: str, n: int, variant: str = 'base') -> CSSCode:
        if code_type not in cls._families:
            raise KeyError(
                f"Unknown code type '{code_type}'. Available: {list(cls._families.keys())}"
            )
        key = (code_type, n, variant)
        if key not in cls._cache:
            cls._cache[key] = cls._families[code_type].load(n, variant)
        return cls._cache[key]

    @classmethod
    def available(cls, code_type: str) -> List[Tuple[int, str]]:
        return cls._families[code_type].available()

    @classmethod
    def iter_all(cls, code_type: str):
        for n, variant in cls.available(code_type):
            yield cls.load(code_type, n, variant)

    @classmethod
    def choices(cls) -> List[str]:
        return list(cls._families.keys())

    @classmethod
    def n_values(cls, code_type: str) -> List[int]:
        return cls._families[code_type].n_values()


for _cls in [TetrahedralCode, TriangularCode]:
    CodeRegistry.register(_cls)
