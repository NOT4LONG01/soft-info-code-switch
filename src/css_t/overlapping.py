"""
overlapping.py
--------------
Sliding-window decoding primitives for CSS memory experiments.

Builds per-window DEM slices via coordinate-based spacetime slicing, threads
physical-error updates (`syn_update`) across window boundaries, and dispatches
to native DEM decoders (full-shot, W ≥ T) or matrix-level decoders (per-window,
W < T) built on top of the `quits` library.

Public API
----------
ler_windowed(code, decoder_name, T, W, p, n_shots, ...)
    Sliding-window LER estimator.  W ≥ T defers to _ler_full_shot (matches
    main.py's memory experiment).  W < T runs the per-window matrix pipeline.

Basis convention
----------------
basis='X'  → X errors on data → Hz detects → project on Lz  (Z-memory)
basis='Z'  → Z errors on data → Hx detects → project on Lx  (X-memory)

Coordinates
-----------
coord[-1] is the round index; requires SHIFT_COORDS per round (emitted by all
circuit builders in circuit.py and by phenomenological_css_circuit).
"""

import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import stim
from scipy.sparse import csc_matrix

from helpers import find_logical_operator, PROJECT_ROOT
from circuit import generate_experiment_with_noise, load_schedule
from noise_models import phenomenological_css_circuit
from leaf_factored import reduce_Lz_leaf_support, leaf_mask
from decoders import build_decoder
from quits.simulation import get_stim_mem_result
from quits.decoder.base import detector_error_model_to_matrix


def _ler_full_shot(circuit, decoder_name, n_shots, seed, max_errors,
                   batch_size, workers=1):
    """Full-shot LER via the native DEM decoder (workers>1 uses sinter.collect)."""
    if workers > 1:
        import sinter
        t0 = time.perf_counter()
        decoder_key = f'_fullshot_{decoder_name}'
        task = sinter.Task(circuit=circuit, decoder=decoder_key)
        stats = sinter.collect(
            num_workers=workers,
            tasks=[task],
            custom_decoders={decoder_key: build_decoder(decoder_name)},
            max_shots=n_shots,
            max_errors=max_errors if max_errors is not None else n_shots,
            max_batch_size=batch_size,
        )
        elapsed = time.perf_counter() - t0
        s = stats[0]
        shots = s.shots
        errors = s.errors
        ler = errors / shots if shots > 0 else 0.0
        dps = elapsed / shots if shots > 0 else 0.0
        return ler, errors, shots, 0.0, dps

    t0 = time.perf_counter()
    dem = circuit.detector_error_model()
    compiled = build_decoder(decoder_name).compile_decoder_for_dem(dem=dem)
    setup_s = time.perf_counter() - t0

    sampler = circuit.compile_detector_sampler(seed=seed)
    total_shots, total_errors = 0, 0
    total_decode_s = 0.0
    while total_shots < n_shots:
        bs = min(batch_size, n_shots - total_shots)
        dets, obs = sampler.sample(
            shots=bs, separate_observables=True, bit_packed=True)

        t0 = time.perf_counter()
        pred = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=dets)
        total_decode_s += time.perf_counter() - t0

        total_errors += int(np.sum((pred != obs).any(axis=1)))
        total_shots += bs
        if max_errors is not None and total_errors >= max_errors:
            break

    decode_s_per_shot = total_decode_s / total_shots if total_shots > 0 else 0.0
    return (total_errors / total_shots, total_errors, total_shots,
            setup_s, decode_s_per_shot)

class UniversalMatrixDecoder:
    """Matrix-level decoder adapter for per-window (quits) sliding-window calls."""
    def __init__(self, check_matrix, decoder_name, channel_probs=None, fallback_p=0.001, **kwargs):
        self.decoder_name = decoder_name
        self.H = check_matrix
        self.core = None
        m_layers = kwargs.get('m', 10)

        if hasattr(self.H, 'toarray'): H_dense = self.H.toarray()
        else: H_dense = np.asarray(self.H)

        if decoder_name in ('bp_osd', 'bposd'):
            from ldpc.bposd_decoder import BpOsdDecoder
            self.core = BpOsdDecoder(self.H, channel_probs=channel_probs, **kwargs)

        elif decoder_name == 'tesseract':
            from tesseract_decoder import tesseract
            dem_lines = []
            # Detector coordinates prevent the A* heuristic from hanging.
            num_detectors = self.H.shape[0]
            for d in range(num_detectors):
                t, i = divmod(d, m_layers)
                dem_lines.append(f"detector({i}, 0, {t}) D{d}")
            for j in range(self.H.shape[1]):
                dets = np.where(H_dense[:, j])[0]
                p = channel_probs[j] if channel_probs is not None else fallback_p
                target_str = " ".join([f"D{d}" for d in dets])
                dem_lines.append(f"error({p}) {target_str}")
            dem = stim.DetectorErrorModel("\n".join(dem_lines))
            self.core = tesseract.TesseractConfig(dem=dem).compile_decoder()

        elif decoder_name == 'mwpf':
            from mwpf import HyperEdge, SolverInitializer, SolverSerialJointSingleHair
            vertex_num = self.H.shape[0]
            weighted_edges = []
            for j in range(self.H.shape[1]):
                dets = np.where(H_dense[:, j])[0]
                p = channel_probs[j] if channel_probs is not None else fallback_p
                # MWPF expects integer weights; -100*log(p) is a standard scaling.
                weight = int(max(1, -100 * np.log(p / (1 - p))))
                weighted_edges.append(HyperEdge(list(dets), weight))
            self.core = SolverSerialJointSingleHair(SolverInitializer(vertex_num, weighted_edges))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        if self.decoder_name == 'tesseract':
            try:
                self.core.decode_to_errors(syndrome.astype(bool))
                indices = self.core.predicted_errors_buffer
                # Tesseract edge indices map 1-to-1 to our error instructions; do not offset.
                res = np.zeros(self.H.shape[1], dtype=int)
                if len(indices) > 0:
                    res[list(indices)] = 1
                return res
            except Exception as e:
                raise RuntimeError(f"Tesseract decoding failed: {e}")

        elif self.decoder_name == 'mwpf':
            from mwpf import SyndromePattern
            self.core.solve(SyndromePattern(np.where(syndrome)[0].tolist()))
            indices = self.core.subgraph()
            res = np.zeros(self.H.shape[1], dtype=int)
            if len(indices) > 0:
                res[list(indices)] = 1
            return res

        return self.core.decode(syndrome)


def _find_schedule(code):
    variant = code.variant or 'base'
    variant_suffix = f'_{variant}' if variant != 'base' else ''
    sched_dir = os.path.join(PROJECT_ROOT, 'data', 'schedule', code.code_type)
    for name in [
        f'sched_{code.code_type}_val{code.n}{variant_suffix}_bposd.json',
        f'sched_{code.code_type}_val{code.n}{variant_suffix}.json',
    ]:
        path = os.path.join(sched_dir, name)
        if os.path.exists(path):
            return load_schedule(path)
    return None


def _group_detectors_by_round(circuit):
    """Map round t → sorted detector ids (coord[-1] as time index)."""
    coords = circuit.get_detector_coordinates()
    round_det = defaultdict(list)
    for det_id, c in coords.items():
        round_det[int(round(c[-1]))].append(det_id)
    for t in round_det:
        round_det[t].sort()
    return dict(round_det)


def _build_window_decoders(circuit, W, F, decoder_kwargs):
    """Coord-based spacetime slicer + per-window decoder construction.

    Each window dict carries: check (DEM slice), obs (observable slice for
    committed part), priors, update (decoded-errors → next-window first-round
    flips; None for last window), det_idx, first_round_n.
    """
    dem = circuit.detector_error_model(decompose_errors=False)
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(dem)
    check_csr = check_matrix.tocsr()

    round_det = _group_detectors_by_round(circuit)
    num_rounds_total = max(round_det.keys()) + 1
    num_rounds = num_rounds_total - 2

    if 2 + num_rounds - W >= 0:
        num_cor_rounds = (2 + num_rounds - W) // F
        if (2 + num_rounds - W) % F != 0:
            num_cor_rounds += 1
    else:
        num_cor_rounds = 0
        warnings.warn("Window size exceeds rounds: doing whole-history correction")

    def _concat(rounds_range):
        return [d for r in rounds_range for d in round_det.get(r, [])]

    def _col_max(sparse_rows):
        col_sum = np.asarray(sparse_rows.sum(axis=0)).flatten()
        nz = np.nonzero(col_sum)[0]
        if nz.size == 0:
            raise ValueError("empty DEM slice — window has no error columns")
        return int(nz.max())

    windows = []
    col_min = 0
    for k in range(num_cor_rounds):
        t_start = F * k
        window_det_idx = _concat(range(t_start, t_start + W))
        commit_det_idx = _concat(range(t_start, t_start + F))

        W_rows = check_csr[window_det_idx, col_min:]
        col_max = _col_max(W_rows)
        W_rows = W_rows[:, :col_max + 1]

        F_corr = W_rows[:len(commit_det_idx), :]
        cor_max = _col_max(F_corr)

        # Priors cover all window columns; obs/update only the committed slice.
        W_priors = priors[col_min : col_min + col_max + 1]
        W_obs = observable_matrix[:, col_min : col_min + cor_max + 1]
        next_round = round_det.get(t_start + F, [])
        W_update = check_csr[next_round, col_min : col_min + cor_max + 1]

        windows.append({
            'check': csc_matrix(W_rows),
            'obs': W_obs,
            'priors': W_priors,
            'update': W_update,
            'det_idx': np.asarray(window_det_idx, dtype=np.int64),
            'first_round_n': len(round_det[t_start]),
        })
        col_min += cor_max + 1

    t_start = F * num_cor_rounds
    last_det_idx = _concat(range(t_start, num_rounds_total))
    W_rows = check_csr[last_det_idx, col_min:]
    windows.append({
        'check': csc_matrix(W_rows),
        'obs': observable_matrix[:, col_min:],
        'priors': priors[col_min:],
        'update': None,
        'det_idx': np.asarray(last_det_idx, dtype=np.int64),
        'first_round_n': len(round_det.get(t_start, [])),
    })

    decoders = []
    for w in windows:
        kw = decoder_kwargs.copy()
        kw['channel_probs'] = w['priors']
        decoders.append(UniversalMatrixDecoder(w['check'], **kw))

    return decoders, windows, num_cor_rounds


def _decode_trials(det_events, decoders, windows, num_cor_rounds):
    num_trials = det_events.shape[0]
    num_obs = windows[0]['obs'].shape[0]
    logical_pred = np.zeros((num_trials, num_obs), dtype=int)

    for i in range(num_trials):
        accumulated = np.zeros(num_obs, dtype=int)
        syn_update = None

        for k in range(num_cor_rounds):
            w = windows[k]
            diff = det_events[i, w['det_idx']].astype(int) % 2
            if syn_update is not None and w['first_round_n'] > 0:
                diff[:w['first_round_n']] = (diff[:w['first_round_n']] + syn_update) % 2
            obs_cols = w['obs'].shape[1]
            decoded = decoders[k].decode(diff)[:obs_cols]
            accumulated = (accumulated + np.asarray(w['obs'] @ decoded).flatten() % 2) % 2
            syn_update = np.asarray(w['update'] @ decoded).flatten() % 2

        w = windows[num_cor_rounds]
        diff = det_events[i, w['det_idx']].astype(int) % 2
        if syn_update is not None and w['first_round_n'] > 0:
            diff[:w['first_round_n']] = (diff[:w['first_round_n']] + syn_update) % 2
        decoded = decoders[num_cor_rounds].decode(diff)
        accumulated = (accumulated + np.asarray(w['obs'] @ decoded).flatten() % 2) % 2
        logical_pred[i, :] = accumulated

    return logical_pred


def ler_windowed(code, decoder_name, T, W, p, n_shots, *,
                 noise_model='phenomenological', basis='Z', stride=None,
                 seed=42, max_errors=None, batch_size=1000,
                 leaf_reduce=False, verbose=False, workers=1):
    """Sliding-window LER estimator.

    Returns (ler, errors, shots, setup_s, decode_s_per_shot).  setup_s is the
    one-time DEM slice + decoder-build cost; decode_s_per_shot is the marginal
    decode-loop cost per shot (sampling excluded).

    leaf_reduce: phenomenological only — substitute a stabilizer-equivalent
    logical op with minimized leaf-qubit support (see leaf_factored).
    """
    if stride is None: stride = 1
    if basis == 'X':
        H_dec = code.Hz
        L_canon = find_logical_operator(code.Hx, code.Hz, basis='Z')
    elif basis == 'Z':
        H_dec = code.Hx
        L_canon = find_logical_operator(code.Hx, code.Hz, basis='X')
    else:
        raise ValueError(f"basis must be 'X' or 'Z'; got {basis!r}")

    if leaf_reduce and noise_model == 'phenomenological':
        L_used = reduce_Lz_leaf_support(H_dec, L_canon, allow_weight_growth=False)
        if verbose:
            leaves = leaf_mask(H_dec)
            print(f"  Lz canonical: supp={list(np.where(L_canon)[0])} "
                  f"wt={int(L_canon.sum())} leaf={int((L_canon & leaves).sum())}")
            print(f"  Lz reduced:   supp={list(np.where(L_used)[0])} "
                  f"wt={int(L_used.sum())} leaf={int((L_used & leaves).sum())}")
    else:
        L_used = L_canon
    L_dec = np.atleast_2d(L_used)

    if noise_model == 'depolarizing':
        if leaf_reduce:
            raise ValueError("leaf_reduce is only supported for phenomenological noise")
        circuit = generate_experiment_with_noise(
            Hx=code.Hx, Hz=code.Hz, rounds=T, noise_model_name='depolarizing',
            noise_params={'p': p, 'p_meas': p}, memory_basis=basis,
            code_type=code.code_type, schedule=_find_schedule(code),
        )
    else:
        circuit = phenomenological_css_circuit(
            H=H_dec, L=L_dec.astype(np.uint8), T=T,
            p_data=p, p_meas=p, noise=basis,
        )

    # W ≥ T: bypass the matrix pipeline and run the full-shot DEM decoder.
    if W >= T:
        return _ler_full_shot(circuit, decoder_name, n_shots, seed,
                              max_errors, batch_size, workers=workers)

    decoder_kwargs = {
        'decoder_name': decoder_name,
        'max_iter': 20,
        'osd_order': min(10, H_dec.shape[1]),
        'fallback_p': p,
        'm': H_dec.shape[0],
    }

    t0 = time.perf_counter()
    decoders, windows, num_cor_rounds = _build_window_decoders(
        circuit, W, stride, decoder_kwargs,
    )
    setup_s = time.perf_counter() - t0

    total_shots, total_errors = 0, 0
    total_decode_s = 0.0
    while total_shots < n_shots:
        bs = min(batch_size, n_shots - total_shots)
        det_events, obs_flips = get_stim_mem_result(circuit, num_trials=bs, seed=seed + total_shots)

        t0 = time.perf_counter()
        logical_pred = _decode_trials(det_events, decoders, windows, num_cor_rounds)
        total_decode_s += time.perf_counter() - t0

        total_errors += int(np.sum((obs_flips != logical_pred).any(axis=1)))
        total_shots += bs
        if max_errors is not None and total_errors >= max_errors: break

    decode_s_per_shot = total_decode_s / total_shots if total_shots > 0 else 0.0
    return total_errors / total_shots, total_errors, total_shots, setup_s, decode_s_per_shot