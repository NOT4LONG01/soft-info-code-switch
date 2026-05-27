"""
soft_info.pipeline.optimize_schedule
------------------------------------
Optimizes syndrome measurement circuit schedules for QEC codes using the
AlphaSyndrome framework (arXiv:2601.12509).

Schedulers
----------
alpha (default)
    Monte Carlo Tree Search (MCTS) guided by a decoder-aware cost function
    (AlphaScheduler).  Decoder-aware: the chosen decoder is called on sampled
    syndromes to evaluate each candidate schedule.  Supports all decoders in
    decoders.py; pymatching and fusion_blossom are handled as native sinter
    decoders while mwpf / bp_osd / tesseract / relay_bp use custom wrappers.

baseline
    Depth-optimal ILP solver (BaselineScheduler).  Decoder-agnostic.

Output files
------------
For each optimised code two files are written to pcms/<code_type>_codes/:

  <code_type>_n{n}[_variant]_alpha-<decoder>.json
      Full AlphaSyndrome schedule in asyndrome's native JSON format.

  sched_<code_type>_val{n}[_variant].json
      Compact schedule in circuit.py's dict format:
          {"ancilla_row,data_qubit_idx": tick_priority, ...}
      Loaded by load_schedule() in circuit.py.  Presence of this file signals
      that optimization is complete; run_optimization() is a no-op if it exists.

  memo_<code_type>_n{n}[_variant]_alpha-<decoder>.txt
      Human-readable summary: code parameters, error model, logical error rates,
      circuit depth, and file paths.

Monkey-patch
------------
DecoderAgent.simulate is replaced with _simulate_with_all_decoders so that
AlphaScheduler can evaluate schedules with any decoder in this project.

Usage
-----
    python optimize_schedule.py go03_self_dual 12
    python optimize_schedule.py go03_self_dual --all --decoder mwpf
    python optimize_schedule.py go03_self_dual --n-list 12 22 32
    python optimize_schedule.py ja25_transversal_t 15 --method baseline
    python optimize_schedule.py qr_dual_containing 7 --decoder mwpf --iters 400
"""

import os
import sys
import json
import argparse
from datetime import datetime
from functools import partial
from typing import Optional, Tuple
import numpy as np

# Ensure DecoderAgent's spawned subprocesses can `import soft_info...`
_PKG_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_existing_pp = os.environ.get("PYTHONPATH", "")
if _PKG_PARENT_DIR not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _PKG_PARENT_DIR + (os.pathsep + _existing_pp if _existing_pp else "")

from ..helpers import find_logical_operator, PROJECT_ROOT
from ..codes.registry import CodeRegistry, _compute_k, schedule_dir
from asyndrome import (
    CSSCode,
    AlphaScheduler,
    BaselineScheduler,
    TrivialModel,
    Schedule,
)
from asyndrome.stimcirc import DecoderAgent, _sinter_predict_observable
from ..decoders.sinter import build_decoder, RELAY_PARAMS
import relay_bp.stim as _relay_bp_stim


_NATIVE_DECODERS = {"pymatching", "fusion_blossom"}  # use sinter.predict_observables directly
_RELAY_DECODERS  = {"relay_bp"}                       # cannot pass through asyndrome pool
_SCHED_MAX_ERRORS: Optional[int] = None               # set >0 for MCTS early stopping


def _simulate_with_all_decoders(self, circuit, nshots: int = 10000):
    custom_decoders = {
        "mwpf":      build_decoder("mwpf"),
        "bp_osd":    build_decoder("bp_osd"),
        "tesseract": build_decoder("tesseract"),
    }

    try:
        dem = circuit._circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )
    except Exception:
        return nshots  # penalise high-weight-check schedules

    sampler = circuit._circuit.compile_detector_sampler()

    # relay_bp falls back to mwpf inside asyndrome's worker pool
    eval_decoder = "mwpf" if self._decoder in _RELAY_DECODERS else self._decoder

    decode = partial(
        _sinter_predict_observable,
        dem=dem,
        decoder=eval_decoder,
        custom_decoders=custom_decoders,
    )

    max_errors = _SCHED_MAX_ERRORS
    if max_errors is None:
        detection_events, observable_flips = sampler.sample(nshots, separate_observables=True)
        if eval_decoder in _NATIVE_DECODERS:
            predictions = decode(detection_events)
        else:
            predictions = self._request(decode, detection_events)
        return int(np.sum(np.any(predictions != observable_flips, axis=1)))
    else:
        # Adaptive early stopping
        batch_size = max(max_errors * 10, 100)
        total_errors = 0
        shots_done = 0
        while shots_done < nshots:
            this_batch = min(batch_size, nshots - shots_done)
            det_events, obs_flips = sampler.sample(this_batch, separate_observables=True)
            if eval_decoder in _NATIVE_DECODERS:
                preds = decode(det_events)
            else:
                preds = self._request(decode, det_events)
            total_errors += int(np.sum(np.any(preds != obs_flips, axis=1)))
            shots_done += this_batch
            if total_errors >= max_errors:
                break
        return total_errors


DecoderAgent.simulate = _simulate_with_all_decoders


def _to_pauli_string(bits: np.ndarray, pauli: str) -> str:
    return "".join(pauli if b else "I" for b in bits)


def build_css_code(
    Hx: np.ndarray,
    Hz: np.ndarray,
    d: int,
    family: str,
) -> CSSCode:
    n = Hx.shape[1]
    k = _compute_k(Hx, Hz)

    x_stabilizers = [_to_pauli_string(row, "X") for row in Hx]
    z_stabilizers = [_to_pauli_string(row, "Z") for row in Hz]

    lx = find_logical_operator(Hx, Hz, basis="X")
    lz = find_logical_operator(Hx, Hz, basis="Z")

    logical_xs = [_to_pauli_string(lx, "X")]
    logical_zs = [_to_pauli_string(lz, "Z")]

    return CSSCode(
        family=family,
        n=n,
        k=max(k, 1),  # guard rank miscounts in degenerate cases
        d=d,
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        logical_xs=logical_xs,
        logical_zs=logical_zs,
    )


def _default_output_dir(code_type: str) -> str:
    return schedule_dir(code_type)


def _circuit_schedule_path(output_dir: str, code_type: str, n: int, variant: str = None, decoder: str = None) -> str:
    variant_suffix = f"_{variant}" if variant and variant != "base" else ""
    decoder_suffix = f"_{decoder}" if decoder else ""
    return os.path.join(output_dir, f"sched_{code_type}_val{n}{variant_suffix}{decoder_suffix}.json")


def run_optimization(
    code_type: str,
    n: int,
    method: str = "alpha",
    decoder: str = "bp_osd",
    p_eval: float = 5e-3,
    iters_per_step: int = 200,
    nshots: int = 1000,
    output_dir: Optional[str] = None,
    variant: Optional[str] = None,
) -> Tuple[Optional[Schedule], Optional[CSSCode]]:
    if output_dir is None:
        output_dir = _default_output_dir(code_type)

    circuit_path = _circuit_schedule_path(output_dir, code_type, n, variant=variant, decoder=decoder)
    if os.path.exists(circuit_path):
        print(f"Schedule already exists for n={n} variant={variant!r} decoder={decoder!r}, skipping → {circuit_path}")
        return None, None

    loaded = CodeRegistry.load(code_type, n, variant=variant or 'base')
    Hx, Hz, d = loaded.Hx, loaded.Hz, loaded.d

    code = build_css_code(Hx, Hz, d=d, family=code_type)
    error_model = TrivialModel(idle_err=p_eval, cnot_err=p_eval)

    print(f"Code  : [[{code.n}, {code.k}, {code.d}]] {code_type}")
    print(f"Method: {method}  |  Decoder: {decoder}")
    print(f"p_eval: {p_eval:.2e}  (idle + cnot error rate for MCTS)")

    if method == "alpha":
        scheduler = AlphaScheduler(iters_per_step=iters_per_step, nshots=nshots)
    elif method == "baseline":
        scheduler = BaselineScheduler()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'alpha' or 'baseline'.")

    schedule: Schedule = scheduler.schedule(code, decoder, error_model)
    print(f"\nOptimised circuit depth (ticks): {schedule.max_tick}")

    os.makedirs(output_dir, exist_ok=True)
    variant_suffix = f"_{variant}" if variant and variant != "base" else ""
    out_path = os.path.join(
        output_dir, f"{code_type}_n{n}{variant_suffix}_{method}-{decoder}.json"
    )
    schedule.to_file(out_path)
    print(f"Schedule saved → {out_path}")

    # (ancilla_row, data_qubit) -> tick_priority
    circuit_schedule = {
        f"{chk.ancilla - code.n},{chk.data}": tick_idx
        for tick_idx, checks_at_tick in enumerate(schedule.checks)
        for chk in checks_at_tick
    }
    circuit_path = _circuit_schedule_path(output_dir, code_type, n, variant=variant, decoder=decoder)
    with open(circuit_path, "w") as f:
        json.dump(circuit_schedule, f)
    print(f"Circuit schedule saved → {circuit_path}")

    print(f"\nEvaluating with {nshots} shots …")
    x_rate, z_rate = schedule.evaluate(code, decoder, error_model, nshots)
    overall = 1.0 - (1.0 - x_rate) * (1.0 - z_rate)
    print(f"  X logical error rate : {x_rate:.4e}")
    print(f"  Z logical error rate : {z_rate:.4e}")
    print(f"  Overall error rate   : {overall:.4e}")

    memo_path = os.path.join(output_dir, f"memo_{code_type}_n{n}{variant_suffix}_{method}-{decoder}.txt")
    with open(memo_path, "w") as f:
        f.write(f"Schedule Optimization Memo\n")
        f.write(f"==========================\n")
        f.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Code       : [[{code.n}, {code.k}, {code.d}]]  ({code_type})\n")
        f.write(f"Method     : {method}\n")
        f.write(f"Decoder    : {decoder}\n")
        f.write(f"Error model: idle={p_eval:.2e}, cnot={p_eval:.2e}\n")
        f.write(f"Shots      : {nshots}\n")
        f.write(f"\nResults\n-------\n")
        f.write(f"Circuit depth (ticks) : {schedule.max_tick}\n")
        f.write(f"X logical error rate  : {x_rate:.4e}\n")
        f.write(f"Z logical error rate  : {z_rate:.4e}\n")
        f.write(f"Overall error rate    : {overall:.4e}\n")
        f.write(f"\nFiles\n-----\n")
        f.write(f"Schedule (asyndrome) : {out_path}\n")
        f.write(f"Schedule (circuit)   : {circuit_path}\n")
    print(f"Memo saved → {memo_path}")

    return schedule, code


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize syndrome measurement circuit schedules for QEC codes.\n"
            "Uses AlphaSyndrome (MCTS, arXiv:2601.12509) or a depth-optimal ILP baseline.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python optimize_schedule.py go03_self_dual 12\n"
            "  python optimize_schedule.py go03_self_dual --all\n"
            "  python optimize_schedule.py go03_self_dual --n-list 12 22 32\n"
            "  python optimize_schedule.py ja25_transversal_t --all --method baseline\n"
            "  python optimize_schedule.py qr_dual_containing 7 --decoder mwpf --iters 400\n"
        ),
    )
    parser.add_argument("code_type", choices=CodeRegistry.choices(), help="QEC code family")
    parser.add_argument("n", type=int, nargs="?", default=None,
                        help="Number of physical qubits (omit with --all or --n-list)")
    n_group = parser.add_mutually_exclusive_group()
    n_group.add_argument("--all", action="store_true",
                         help="Run for all n values defined in the code config")
    n_group.add_argument("--n-list", type=int, nargs="+", metavar="N",
                         help="Run for specific n values, e.g. --n-list 12 22 32")
    parser.add_argument("--method", choices=["alpha", "baseline"], default="alpha",
                        help="Scheduler: 'alpha' MCTS (default) or 'baseline' ILP")
    parser.add_argument("--decoder",
                        choices=["pymatching", "fusion_blossom", "bp_osd", "mwpf", "tesseract", "relay_bp"],
                        default="bp_osd", help="Decoder used during schedule search (default: bp_osd)")
    parser.add_argument("--p-eval", type=float, default=5e-3, metavar="P",
                        help="Physical error rate used during MCTS evaluation (default: 5e-3). "
                             "Must be high enough that logical errors occur within nshots, "
                             "but below threshold so good schedules are rewarded.")
    parser.add_argument("--iters", type=int, default=200, dest="iters_per_step", metavar="N",
                        help="MCTS iterations per scheduling step (default: 200)")
    parser.add_argument("--nshots", type=int, default=1000, metavar="N",
                        help="Monte Carlo shots per evaluation (default: 1000)")
    parser.add_argument("--max-sched-errors", type=int, default=None, metavar="N",
                        help="Stop each MCTS simulation once N logical errors are observed "
                             "(analogous to sinter max_errors; default: disabled)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Output directory for schedule JSON files")
    parser.add_argument("--variant", default=None, metavar="VARIANT",
                        help="Code variant for JA25 (e.g. rm, rm2, tetra, gauge_qHam, base)")
    args = parser.parse_args()

    if args.all:
        n_values = CodeRegistry.n_values(args.code_type)
    elif args.n_list:
        n_values = args.n_list
    elif args.n is not None:
        n_values = [args.n]
    else:
        parser.error("Provide a positional n, --all, or --n-list.")

    global _SCHED_MAX_ERRORS
    _SCHED_MAX_ERRORS = args.max_sched_errors

    failed = []
    for i, n in enumerate(n_values):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(n_values)}] Optimizing n={n}")
        print(f"{'='*60}")
        try:
            run_optimization(
                code_type=args.code_type,
                n=n,
                method=args.method,
                decoder=args.decoder,
                p_eval=args.p_eval,
                iters_per_step=args.iters_per_step,
                nshots=args.nshots,
                output_dir=args.output_dir,
                variant=args.variant,
            )
        except Exception as e:
            print(f"  ERROR for n={n}: {e}")
            failed.append((n, e))

    if failed:
        print(f"\nFailed for n values: {[n for n, _ in failed]}")
    else:
        print(f"\nAll {len(n_values)} sizes completed successfully.")


if __name__ == "__main__":
    main()
