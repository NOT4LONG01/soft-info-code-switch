"""
soft_info.pipeline.main
-----------------------
CLI entry point for the QEC decoding pipeline (``python -m soft_info.pipeline.main``).

Loads codes via CodeRegistry, builds noisy Stim circuits, runs sinter
sampling, and writes per-rank CSV results.

Arguments
---------
--code_type     Code family (see CodeRegistry.choices()).
--n             Single n value; default runs all n for the code type.
--variant       Code variant (e.g. rm, rm2, tetra, gauge_qHam).
--decoder       mwpf | tesseract | bp_osd | relay_bp | all (default: mwpf).
--p_values      Space-separated physical error rates to simulate.
--max_shots     Hard ceiling on shots per p-value (default: 100_000_000).
--max_errors    Stop each p-value after this many logical errors (default: 1000).
--workers       Parallel sinter workers; defaults to SLURM_CPUS_PER_TASK or cpu_count.
--circuit       Override circuit topology: color | self_dual | standard.

SLURM
-----
Auto-detects SLURM_PROCID / SLURM_NTASKS and distributes tasks across ranks
by index modulo world_size.

Output
------
data/results/<decoder>/<code_tag>_<noise_model>_<decoder>_rank<N>.csv
"""

import sys
import os
import json
import argparse
import glob
import time
import datetime
import traceback
import hashlib

import numpy as np
import pandas as pd
import sinter

from ..helpers import parse_and_average_stats, PROJECT_ROOT
from ..codes import CodeRegistry, schedule_dir
from ..codes.circuit import generate_experiment_with_noise, load_schedule
from ..decoders.sinter import build_decoder, ALL_DECODERS, EXTRA_DECODERS


def _write_manifest(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _print_summary(summary_df, decoder_name: str, proc_rank: int) -> None:
    if summary_df is None or summary_df.empty:
        return
    cols = ['n', 'd', 'variant', 'p', 'shots', 'errors',
            'total_logical_error_rate', 'average_cpu_time_seconds']
    for c in cols:
        if c not in summary_df.columns:
            summary_df[c] = 0

    w = 96
    print(f"\n{'='*w}")
    print(f"  RESULTS  decoder={decoder_name}  rank={proc_rank}  "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*w}")
    hdr = f"{'n':>6}  {'d':>4}  {'variant':>10}  {'p':>9}  "
    hdr += f"{'shots':>10}  {'errors':>8}  {'p_L':>10}  {'cpu_ms':>9}"
    print(hdr)
    print(f"{'-'*w}")
    for _, row in summary_df.sort_values(['n', 'p']).iterrows():
        cpu_s = row['average_cpu_time_seconds']
        cpu_str = f"{float(cpu_s)*1000:>9.3f}" if cpu_s is not None and cpu_s == cpu_s else f"{'N/A':>9}"
        print(
            f"{int(row['n']):>6}  {int(row['d']):>4}  "
            f"{str(row.get('variant','base')):>10}  "
            f"{float(row['p']):>9.2e}  {int(row['shots']):>10}  "
            f"{int(row['errors']):>8}  "
            f"{float(row['total_logical_error_rate']):>10.2e}  "
            f"{cpu_str}"
        )
    print(f"{'='*w}\n")


def main():
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    is_slurm = slurm_cpus is not None

    parser = argparse.ArgumentParser()
    parser.add_argument("--code_type", type=str, required=True,
                        choices=CodeRegistry.choices())
    parser.add_argument("--n", type=int, default=None,
                        help="Run only this n value (default: all n for the code type)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Code variant (e.g. rm, rm2, tetra, gauge_qHam)")
    parser.add_argument("--decoder", type=str, default="mwpf",
                        choices=ALL_DECODERS + EXTRA_DECODERS + ["all"],
                        help="Decoder to use, or 'all' to run mwpf+tesseract")
    parser.add_argument("--max_shots", type=int, default=100_000_000)
    parser.add_argument("--max_errors", type=int, default=1000,
                        help="Stop each p-value task after this many logical errors")
    parser.add_argument("--p_values", type=float, nargs="+",
                        required=True,
                        help="Physical error rates to simulate")
    parser.add_argument("--workers", type=int,
                        default=int(slurm_cpus) if is_slurm else os.cpu_count())
    parser.add_argument("--circuit", type=str, default=None,
                        choices=["color", "self_dual", "standard"],
                        help="Override circuit topology: color (C_XYZ), self_dual, or standard CSS")
    args = parser.parse_args()

    selected_decoders = ALL_DECODERS if args.decoder == "all" else [args.decoder]

    proc_rank  = int(os.environ.get("SLURM_PROCID",  0))
    world_size = int(os.environ.get("SLURM_NTASKS",  1))

    iter_list    = [args.n] if args.n is not None else CodeRegistry.n_values(args.code_type)
    noise_values = sorted(args.p_values, reverse=True)
    model_name   = "depolarizing"

    data_dir  = os.path.join(PROJECT_ROOT, "data")
    sched_dir = schedule_dir(args.code_type)

    if proc_rank == 0:
        os.makedirs(sched_dir, exist_ok=True)
    time.sleep(2)  # allow rank 0 to finish mkdir before other ranks proceed

    n_tag       = f"_n{args.n}" if args.n is not None else ""
    variant_tag = f"_{args.variant}" if args.variant and args.variant != "base" else ""
    circuit_tag = f"_{args.circuit}" if args.circuit else ""
    base_name   = f"{args.code_type}{n_tag}{variant_tag}{circuit_tag}"

    if proc_rank == 0:
        print(f"\n{'='*60}")
        print(f"  QEC Decoding Pipeline")
        print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"  code_type : {args.code_type}")
        print(f"  n values  : {iter_list}")
        print(f"  variant   : {args.variant or 'base'}")
        print(f"  circuit   : {args.circuit or 'default'}")
        print(f"  decoders  : {selected_decoders}")
        print(f"  max_shots : {args.max_shots:,}")
        print(f"  workers   : {args.workers}  (ranks: {world_size})")
        print(f"  noise pts : {len(noise_values)}  [{noise_values[0]:.1e} .. {noise_values[-1]:.1e}]")
        print(f"{'='*60}\n")

    for decoder_name in selected_decoders:
        results_dir = os.path.join(data_dir, "results", decoder_name)
        tmp_dir     = os.path.join(data_dir, "tmp",     decoder_name)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(tmp_dir,     exist_ok=True)

        resume_path   = os.path.join(tmp_dir,     f"resume_{base_name}_{model_name}_{decoder_name}_rank{proc_rank}.sinter")
        part_filename = os.path.join(results_dir, f"{base_name}_{model_name}_{decoder_name}_rank{proc_rank}.csv")
        manifest_path = os.path.join(results_dir, f"manifest_{decoder_name}_rank{proc_rank}.json")

        # Skip p-values already at max_shots with <10 errors (resume logic)
        saturated_ps: set = set()
        existing_data = []
        if os.path.exists(resume_path):
            try:
                existing_data = sinter.stats_from_csv_files(resume_path)
                if existing_data and proc_rank == 0:
                    print(f"  Resuming from {len(existing_data)} existing records.", flush=True)
                p_agg: dict = {}
                for s in existing_data:
                    p = (s.json_metadata or {}).get('p')
                    if p is None:
                        continue
                    if p not in p_agg:
                        p_agg[p] = {'shots': 0, 'errors': 0}
                    p_agg[p]['shots']  += s.shots
                    p_agg[p]['errors'] += s.errors
                for p, acc in p_agg.items():
                    if acc['shots'] >= args.max_shots and acc['errors'] < 10:
                        saturated_ps.add(float(p))
                if saturated_ps and proc_rank == 0:
                    skipped = sorted(saturated_ps)
                    print(f"  [SKIP] {len(skipped)} p-value(s) already at max_shots with <10 errors: "
                          f"{[f'{v:.2e}' for v in skipped]}", flush=True)
            except Exception:
                pass

        active_noise_values = [p for p in noise_values if float(p) not in saturated_ps]
        if not active_noise_values:
            if proc_rank == 0:
                print(f"  All p-values saturated for {decoder_name} — skipping.", flush=True)
            continue

        base_task_defs = []
        for val in iter_list:
            try:
                code = CodeRegistry.load(args.code_type, val,
                                         variant=args.variant or 'base')
                variant_suffix = f"_{code.variant}" if code.variant and code.variant != "base" else ""
                schedule_file = os.path.join(
                    sched_dir, f"sched_{args.code_type}_val{val}{variant_suffix}_{decoder_name}.json")
                if not os.path.exists(schedule_file):
                    schedule_file = os.path.join(
                        sched_dir, f"sched_{args.code_type}_val{val}{variant_suffix}.json")
                schedule = load_schedule(schedule_file) if os.path.exists(schedule_file) else {}
                for p in active_noise_values:
                    circuit = generate_experiment_with_noise(
                        code.Hx, code.Hz, code.d * 3, model_name, {"p": p},
                        schedule=schedule, code_type=args.code_type,
                        circuit_type=args.circuit,
                    )
                    base_task_defs.append((circuit, {
                        'n': code.n, 'd': code.d, 'r': code.d * 3, 'p': p,
                        'noise_model': model_name, 'code_type': args.code_type,
                        'iter_val': val, 'variant': code.variant,
                        'circuit': args.circuit or 'default',
                    }))
            except Exception as e:
                if proc_rank == 0:
                    print(f"  [SKIP] val={val}: {e}")

        if not base_task_defs:
            if proc_rank == 0:
                print(f"  No valid codes loaded for {decoder_name} — skipping.", flush=True)
            continue

        all_tasks = [
            sinter.Task(
                circuit=circuit,
                decoder=decoder_name,
                json_metadata={**meta, 'decoder': decoder_name},
            )
            for circuit, meta in base_task_defs
        ]

        # SLURM rank distribution: each rank takes tasks i % world_size == proc_rank
        my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == proc_rank]
        if not my_tasks:
            continue

        # Unique decoder key + trace file per task for isolation
        custom_decoders = {}
        for task in my_tasks:
            meta = task.json_metadata
            uid_key = f"{decoder_name}_n{meta['n']}_p{meta['p']:.6e}_{args.code_type}_{meta.get('variant','base')}_rank{proc_rank}"
            uid = f"{decoder_name}_{hashlib.sha256(uid_key.encode()).hexdigest()[:12]}"
            trace_path = os.path.join(tmp_dir, f"trace_{uid}.bin")
            task.decoder = uid
            task.json_metadata['trace_path'] = trace_path
            custom_decoders[uid] = build_decoder(decoder_name, trace_filename=trace_path)

        manifest = {
            'start_time': datetime.datetime.now().isoformat(),
            'code_type': args.code_type,
            'n_values': [int(v) for v in iter_list],
            'variant': args.variant or 'base',
            'decoder': decoder_name,
            'max_shots': args.max_shots,
            'workers': args.workers,
            'n_tasks': len(my_tasks),
            'proc_rank': proc_rank,
            'world_size': world_size,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_nodeid': os.environ.get('SLURM_NODEID'),
            'status': 'running',
        }
        _write_manifest(manifest_path, manifest)
        print(f"[rank {proc_rank}] {decoder_name}: {len(my_tasks)} tasks  →  {results_dir}", flush=True)

        # Precompute strong_ids for stale-data filter
        task_strong_ids = set()
        for task in my_tasks:
            try:
                dem = task.circuit.detector_error_model()
            except Exception:
                try:
                    dem = task.circuit.detector_error_model(ignore_decomposition_failures=True)
                except Exception:
                    continue
            tmp = sinter.Task(circuit=task.circuit, detector_error_model=dem,
                              decoder=task.decoder, json_metadata=task.json_metadata)
            try:
                task_strong_ids.add(tmp.strong_id())
            except Exception:
                pass

        summary_df = None
        try:
            iterator = sinter.iter_collect(
                num_workers=args.workers,
                tasks=my_tasks,
                custom_decoders=custom_decoders,
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                additional_existing_data=existing_data,
                max_batch_seconds=30,
                max_batch_size=1000,
            )

            with open(resume_path, 'a') as resume_file:
                if resume_file.tell() == 0:
                    print(sinter.CSV_HEADER, file=resume_file)

                current_strong_ids = set()
                accumulated: dict = {}
                last_flush_time = time.monotonic()

                for progress in iterator:
                    for stat in progress.new_stats:
                        print(stat.to_csv_line(), file=resume_file, flush=True)
                        current_strong_ids.add(stat.strong_id)
                        p = (stat.json_metadata or {}).get('p')
                        if p not in accumulated:
                            accumulated[p] = {'shots': 0, 'errors': 0}
                        accumulated[p]['shots']  += stat.shots
                        accumulated[p]['errors'] += stat.errors
                        shots  = accumulated[p]['shots']
                        errors = accumulated[p]['errors']
                        print(f"[rank {proc_rank}] {decoder_name}"
                              f"  p={float(p):.2e}  shots={shots:>10}  errors={errors:>6}",
                              flush=True)

                    # Flush partial CSV every 10 minutes
                    if time.monotonic() - last_flush_time >= 600:
                        last_flush_time = time.monotonic()
                        try:
                            partial_stats = sinter.stats_from_csv_files(resume_path)
                            partial_stats = [s for s in partial_stats
                                             if s.strong_id in current_strong_ids]
                            if partial_stats:
                                tmp_df = parse_and_average_stats(partial_stats, model_name)
                                if os.path.exists(part_filename) and os.path.getsize(part_filename) > 1:
                                    try:
                                        existing_df = pd.read_csv(part_filename)
                                        new_ps = set(tmp_df["p"].tolist())
                                        kept = existing_df[~existing_df["p"].isin(new_ps)]
                                        tmp_df = pd.concat([kept, tmp_df], ignore_index=True).sort_values("p")
                                    except Exception:
                                        pass
                                tmp_df.to_csv(part_filename, index=False)
                        except Exception:
                            pass

            for p, acc in sorted(accumulated.items(), key=lambda x: float(x[0])):
                shots  = acc['shots']
                errors = acc['errors']
                p_L    = errors / shots if shots > 0 else 0.0
                print(f"[rank {proc_rank}] {decoder_name}"
                      f"  p={float(p):.2e}  shots={shots:>10}  errors={errors:>6}"
                      f"  p_L={p_L:.3e}  DONE", flush=True)

            full_stats = sinter.stats_from_csv_files(resume_path)
            effective_ids = current_strong_ids or task_strong_ids
            if effective_ids:
                stale = [s for s in full_stats if s.strong_id not in effective_ids]
                if stale:
                    print(f"[rank {proc_rank}] WARNING: dropping {len(stale)} stale records "
                          f"from resume file (circuit has changed). Delete resume file to suppress.",
                          flush=True)
                full_stats = [s for s in full_stats if s.strong_id in effective_ids]

            p_to_trace = {t.json_metadata['p']: t.json_metadata['trace_path']
                          for t in my_tasks if 'trace_path' in t.json_metadata}
            for s in full_stats:
                p = (s.json_metadata or {}).get('p')
                if p in p_to_trace and s.json_metadata is not None:
                    s.json_metadata['trace_path'] = p_to_trace[p]

            summary_df = parse_and_average_stats(full_stats, model_name)
            if os.path.exists(part_filename) and os.path.getsize(part_filename) > 1:
                try:
                    existing_df = pd.read_csv(part_filename)
                    new_ps = set(summary_df["p"].tolist())
                    kept = existing_df[~existing_df["p"].isin(new_ps)]
                    summary_df = pd.concat([kept, summary_df], ignore_index=True).sort_values("p")
                except Exception:
                    pass  # corrupted file — overwrite
            summary_df.to_csv(part_filename, index=False)

            manifest['end_time'] = datetime.datetime.now().isoformat()
            manifest['status'] = 'completed'
            manifest['result_file'] = part_filename
            _write_manifest(manifest_path, manifest)

            _print_summary(summary_df, decoder_name, proc_rank)

            for task in my_tasks:
                tf = task.json_metadata.get('trace_path')
                if tf:
                    for f in glob.glob(f"{tf}*"):
                        try:
                            os.remove(f)
                        except OSError:
                            pass

            print(f"[rank {proc_rank}] Finished {decoder_name}.", flush=True)

        except Exception as e:
            manifest['end_time'] = datetime.datetime.now().isoformat()
            manifest['status'] = 'failed'
            manifest['error'] = str(e)
            _write_manifest(manifest_path, manifest)
            print(f"[rank {proc_rank}] Failed ({decoder_name}): {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
