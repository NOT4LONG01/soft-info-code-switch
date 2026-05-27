"""
leaf_factored.py
----------------
Observable-gauge rewrite that minimizes leaf-qubit support in logical
operators. Used by `overlapping.ler_windowed(leaf_reduce=True, ...)`.

Why it helps
------------
A leaf qubit j has Hz[:, j].sum() == 1 — it appears in exactly one Z-stabilizer
row r(j). In the phenomenological memory circuit, an X error on leaf j at
round t flips exactly one detector: det[r(j), t]. A measurement error on
row r(j) at round t flips det[r(j), t] and det[r(j), t+1]. At boundary
rounds these share a single-flip signature, so leaf X errors and
boundary meas errors are *perfectly degenerate* in the syndrome.

If the canonical Lz has support on leaf qubits, the decoder must guess
whether a lone boundary detector came from a leaf-qubit X error (flips
observable) or a measurement error (doesn't). Rewriting Lz to a
stabilizer-equivalent form with zero leaf support collapses this
ambiguity — both interpretations agree on the observable.

Measured effect at p=0.01, T=14 (phenomenological, tesseract):
    ja25/tetra:     leaf 1→0 @ wt 3→3 → 0.6× LER
    ja25/tetra_opt: leaf 1→0 @ wt 3→3 → 0.78–0.95× LER
    ja25/rm:        leaf 1→0 @ wt 3→3 → 0.66–0.71× LER
    ja25/rm_opt:    canonical already leaf-free → 1.00×
    tetrahedral:    0 leaves → N/A

API
---
    leaf_mask(Hz) -> bool[n]
    reduce_Lz_leaf_support(Hz, Lz, *, allow_weight_growth=False) -> Lz_min
"""

from itertools import combinations

import numpy as np


def leaf_mask(Hz):
    """Boolean mask of leaf qubits (Hz column weight 1)."""
    Hz = np.asarray(Hz, dtype=np.uint8)
    return np.array([Hz[:, j].sum() == 1 for j in range(Hz.shape[1])], dtype=bool)


def reduce_Lz_leaf_support(Hz, Lz, max_rows=None, allow_weight_growth=False):
    """Return a stabilizer-equivalent Lz minimizing leaf-qubit support.

    Searches subsets of Hz rows (up to max_rows) and XORs them into Lz,
    choosing the one with smallest leaf intersection (ties: smallest weight).

    allow_weight_growth=False (default) rejects any candidate heavier than
    the input — avoids pessimal gauges where the heavier observable
    contracts more non-leaf errors than the leaf gauge frees.
    """
    Hz = np.asarray(Hz, dtype=np.uint8)
    Lz = np.asarray(Lz, dtype=np.uint8).copy()
    leaves = leaf_mask(Hz)
    m = Hz.shape[0]
    max_rows = m if max_rows is None else min(max_rows, m)
    orig_wt = int(Lz.sum())

    best_leaf = int((Lz & leaves).sum())
    best_wt = orig_wt
    best_L = Lz.copy()
    for k in range(1, max_rows + 1):
        for combo in combinations(range(m), k):
            L = Lz.copy()
            for r in combo:
                L ^= Hz[r]
            wt = int(L.sum())
            if wt == 0:
                continue
            if not allow_weight_growth and wt > orig_wt:
                continue
            leaf_ct = int((L & leaves).sum())
            if leaf_ct < best_leaf or (leaf_ct == best_leaf and wt < best_wt):
                best_leaf = leaf_ct
                best_wt = wt
                best_L = L.copy()
        if best_leaf == 0:
            break
    return best_L
