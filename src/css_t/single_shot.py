#!/usr/bin/env python3
"""
single_shot.py
--------------
Sweep runner for the single-shot decoding property.  Two modes:

  --mode w  (default)  Fix T, sweep W → LER vs W.
                        Flat tail after some w* ⇒ single-shot at w*.
                        CSV:  wsweep_<tag>_n<N>_T<T>_p<P>_<nm>_s<shots>.csv
  --mode t             Fix W, sweep T → LER vs T.
                        Linear growth ⇒ constant per-round LER ⇒ single-shot.
                        CSV:  tsweep_<tag>_n<N>_W<W>_p<P>_<nm>_s<shots>.csv

Defaults (w-mode): T = 3*d, W = 1..T.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codes import CodeRegistry
from helpers import PROJECT_ROOT
from overlapping import ler_windowed


def _worker(task):
    """One (code, decoder, W, T) evaluation, run in a child process."""
    (code_type, n, variant, decoder_name, W, T, p, n_shots,
     noise_model, basis, stride, seed, max_errors) = task

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from codes import CodeRegistry
    from overlapping import ler_windowed

    code = CodeRegistry.load(code_type, n, variant)
    ler, errs, shots, setup_s, decode_per_shot = ler_windowed(
        code, decoder_name, T, W, p, n_shots,
        noise_model=noise_model, basis=basis, stride=stride,
        seed=seed, max_errors=max_errors,
    )
    return code_type, n, variant, decoder_name, W, T, ler, errs, shots, setup_s, decode_per_shot


def _label(code_type, variant):
    return f"{code_type}/{variant}" if variant else code_type


def main():
    parser = argparse.ArgumentParser(description="Single-shot decoding property sweep runner.")
    parser.add_argument("code_type", nargs="?", default="tetrahedral")
    parser.add_argument("n", nargs="?", type=int, default=15)
    parser.add_argument("--variant",   type=str, default=None)
    parser.add_argument("--decoder",   choices=["mwpf", "tesseract", "bp_osd"], default="bp_osd")
    parser.add_argument("--noise",
                        choices=["phenomenological", "depolarizing"],
                        default="phenomenological")
    parser.add_argument("--basis",     choices=["Z", "X"], default="X")
    parser.add_argument("--p",         type=float, default=0.01)
    parser.add_argument("--mode",      choices=["w", "t"], default="w",
                        help="w: fix T, sweep W; t: fix W, sweep T")
    parser.add_argument("--T",         type=int, default=None,
                        help="(w-mode) Total rounds; defaults to 3*code.d.")
    parser.add_argument("--W-values",  type=str, default="full",
                        help="(w-mode) Space-separated window sizes; 'full' = 1..T.")
    parser.add_argument("--W",         type=int, default=None,
                        help="(t-mode) Fixed window size.")
    parser.add_argument("--T-values",  type=str, default=None,
                        help="(t-mode) Space-separated T values; default W..3*d.")
    parser.add_argument("--stride",    type=int, default=1)
    parser.add_argument("--shots",     type=int, default=1_000_000)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--workers",   type=int, default=os.cpu_count())
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out-dir",   type=str, default=None)
    args = parser.parse_args()

    code = CodeRegistry.load(args.code_type, args.n, args.variant)
    m, n_qubits = code.Hz.shape
    rank = int(np.linalg.matrix_rank(code.Hz))
    meta = m - rank

    if args.mode == "w":
        if args.T is None:
            args.T = 3 * code.d
        if args.W_values.strip().lower() == "full":
            W_values = list(range(1, args.T + 1))
        else:
            W_values = [int(w) for w in args.W_values.split() if w.strip()]
        T_values = [args.T]
        sweep_pairs = [(W, args.T) for W in W_values]
    else:
        if args.W is None:
            parser.error("--W is required in t-mode")
        if args.T_values is None:
            T_values = list(range(args.W, 3 * code.d + 1))
        else:
            T_values = [int(t) for t in args.T_values.split() if t.strip()]
        W_values = [args.W]
        sweep_pairs = [(args.W, T) for T in T_values]

    print(f"\n{'Code':<22} {'m':>4} {'rank':>6} {'meta':>6}")
    print("─" * 44)
    ss = "  ← single-shot" if meta > 0 else ""
    print(f"{_label(args.code_type, args.variant):<22} {m:>4} {rank:>6} {meta:>6}{ss}")
    if args.mode == "w":
        print(f"\n[w-sweep]  T = {args.T} | W = {W_values} | noise = {args.noise} | "
              f"p = {args.p} | decoder = {args.decoder}")
    else:
        print(f"\n[t-sweep]  W = {args.W} | T = {T_values} | noise = {args.noise} | "
              f"p = {args.p} | decoder = {args.decoder}")
    print(f"shots = {args.shots} | max_errors = {args.max_errors} | workers = {args.workers}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data" / "results" / "single_shot"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.code_type, args.n, args.variant, args.decoder, W, T, args.p,
         args.shots, args.noise, args.basis, args.stride,
         args.seed + i, args.max_errors)
        for i, (W, T) in enumerate(sweep_pairs)
    ]

    results = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            _, _, _, dec, W, T, ler, errs, shots, setup_s, dps = fut.result()
            results[(dec, W, T)] = (ler, errs, shots, setup_s, dps)
            dt = time.perf_counter() - t0
            axis = f"w={W:<2d}" if args.mode == "w" else f"T={T:<2d}"
            print(f"  [{i:3d}/{len(tasks)}]  {dec:<10}  {axis}  LER={ler:.5f}  "
                  f"({errs}/{shots})  setup={setup_s*1e3:.1f}ms  decode={dps*1e3:.2f}ms/shot  {dt:.0f}s",
                  flush=True)

    tag = f"{args.code_type}" + (f"_{args.variant}" if args.variant else "")
    nm  = "dep" if args.noise == "depolarizing" else "phenom"
    if args.mode == "w":
        csv_name = f"wsweep_{tag}_n{args.n}_T{args.T}_p{args.p}_{nm}_s{args.shots}.csv"
    else:
        csv_name = f"tsweep_{tag}_n{args.n}_W{args.W}_p{args.p}_{nm}_s{args.shots}.csv"
    csv_path = out_dir / csv_name

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "n", "variant", "decoder", "W", "T", "p", "noise",
                    "ler", "errors", "shots", "setup_s", "decode_s_per_shot"])
        for (dec, W, T), (ler, errs, shots, setup_s, dps) in sorted(results.items()):
            w.writerow([args.code_type, args.n, args.variant or "", dec, W, T, args.p,
                        args.noise, f"{ler:.8f}", errs, shots, f"{setup_s:.6e}", f"{dps:.6e}"])

    print(f"\nsaved → {csv_path}")

if __name__ == "__main__":
    main()