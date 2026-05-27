"""
helpers.py
----------
Project-wide utility functions.

Exports
-------
PROJECT_ROOT
    Absolute path to the repository root (two levels above this file).

find_logical_operator(Hx, Hz, basis='Z') -> np.ndarray
    Finds a representative logical X or Z operator by searching the GF(2) null
    space of the opposite parity-check matrix and selecting a coset representative
    not in the stabilizer row space.  Raises ValueError if none is found.

parse_and_average_stats(stats, model_name) -> pd.DataFrame
    Converts a list of sinter.TaskStats into an aggregated result DataFrame.
    Reads binary trace files (written by decoders._write_trace) to extract
    per-shot CPU timing and, for MWPF, objective lower bounds.
    Output columns: noise_model, n, d, r, p, code_type, variant, decoder,
                    shots, errors, total_logical_error_rate,
                    mean_objective_per_syndrome, mean_primal_dual_gap,
                    gap_nonzero_fraction, average_cpu_time_seconds.
"""

import os
import numpy as np
import pandas as pd
import galois
import sinter

from typing import List

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))


def find_logical_operator(Hx, Hz, basis="Z"):
    F2 = galois.GF(2)
    gf_Hx = F2(Hx.astype(int))
    gf_Hz = F2(Hz.astype(int))

    candidates_basis = gf_Hx.null_space() if basis == "Z" else gf_Hz.null_space()
    stabilizers = gf_Hz if basis == "Z" else gf_Hx
    stab_rank = np.linalg.matrix_rank(stabilizers)

    for cand in candidates_basis:
        combined = np.concatenate((stabilizers, np.atleast_2d(cand)), axis=0)
        if np.linalg.matrix_rank(combined) > stab_rank:
            return np.array(cand, dtype=np.uint8)

    raise ValueError(f"Could not find a logical {basis} operator!")


def parse_and_average_stats(stats: List[sinter.TaskStats], model_name: str) -> pd.DataFrame:
    from decoders import read_trace
    results = []
    for s in stats:
        m = s.json_metadata or {}
        trace_file = m.get('trace_path')
        avg_obj, avg_cpu, avg_gap, gap_frac = None, None, None, None

        decoder_name = m.get('decoder', 'mwpf')
        if trace_file:
            try:
                df_trace = read_trace(trace_file)
                if not df_trace.empty:
                    avg_cpu = float(df_trace['cpu_time'].mean())
                    if decoder_name == 'mwpf':
                        nontrivial = df_trace['obj_upper'] > 1e-9  # obj_upper>0 → non-empty correction
                        n_nontrivial = int(nontrivial.sum())
                        avg_obj = (float(df_trace.loc[nontrivial, 'obj_lower'].mean())
                                   if n_nontrivial > 0 else 0.0)
                        gap = df_trace['obj_upper'] - df_trace['obj_lower']
                        avg_gap = (float(gap[nontrivial].mean())
                                   if n_nontrivial > 0 else 0.0)
                        gap_frac = float((gap > 1e-6).mean())
            except Exception as e:
                print(f"[ERROR] Trace parse failed for {trace_file}: {e}")

        results.append({
            'noise_model': model_name,
            'n': m.get('n'), 'd': m.get('d'), 'r': m.get('r'), 'p': m.get('p'),
            'code_type': m.get('code_type', 'unknown'),
            'variant': m.get('variant', 'base'),
            'decoder': decoder_name,
            'shots': s.shots, 'errors': s.errors,
            'total_logical_error_rate': s.errors / s.shots if s.shots > 0 else 0,
            'mean_objective_per_syndrome': avg_obj,
            'mean_primal_dual_gap': avg_gap if decoder_name == 'mwpf' and trace_file else None,
            'gap_nonzero_fraction': gap_frac if decoder_name == 'mwpf' and trace_file else None,
            'average_cpu_time_seconds': avg_cpu,
        })
    return pd.DataFrame(results)
