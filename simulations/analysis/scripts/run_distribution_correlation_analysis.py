#!/usr/bin/env python
"""
Script to analyze correlation between logical error distribution and fixed-class decoding LLRs.

This script runs the computationally intensive part of the analysis:
1. Loads/computes logical error distribution
2. Selects representative errors across the distribution
3. For each shot: standard decode + fixed-class decodes for selected errors
4. Saves results to a parquet file for analysis in notebook

Usage:
    python run_distribution_correlation_analysis.py --help
    python run_distribution_correlation_analysis.py --n-shots 10000 --output results.parquet
    python run_distribution_correlation_analysis.py --n-shots 50000 --n-errors 20 --p 0.003
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "simulations"))

from ldpc_post_selection.bplsd_decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.logical_error_distribution import (
    index_to_logical_class,
    collect_logical_error_distribution_fast,
)
from simulations.utils.build_circuit import build_BB_circuit


def select_representative_errors(
    distribution: np.ndarray,
    n_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select representative errors uniformly spaced across the sorted distribution.

    Parameters
    ----------
    distribution : 1D numpy array of int
        Logical error distribution where index 0 is correct decoding.
    n_samples : int
        Number of representative errors to select.

    Returns
    -------
    selected_indices : 1D numpy array of int
        Original error indices (1 to 2^k - 1).
    selected_ranks : 1D numpy array of int
        Rank positions (0 = most likely, higher = less likely).
    selected_probs : 1D numpy array of float
        Probability of each selected error (among errors only, excluding correct).
    """
    # Exclude index 0 (correct decoding)
    error_dist = distribution[1:]
    error_indices = np.arange(1, len(distribution))

    # Sort by count (descending)
    sorted_order = np.argsort(error_dist)[::-1]
    sorted_indices = error_indices[sorted_order]
    sorted_counts = error_dist[sorted_order]

    # Total error count (excluding correct)
    total_errors = error_dist.sum()
    sorted_probs = (
        sorted_counts / total_errors
        if total_errors > 0
        else sorted_counts.astype(float)
    )

    # Select uniformly spaced ranks
    n_errors = len(sorted_indices)
    rank_positions = np.linspace(0, n_errors - 1, n_samples, dtype=int)

    selected_indices = sorted_indices[rank_positions]
    selected_ranks = rank_positions
    selected_probs = sorted_probs[rank_positions]

    return selected_indices, selected_ranks, selected_probs


def _process_shots_chunk(
    chunk_start: int,
    chunk_size: int,
    det_chunk: np.ndarray,
    circuit,
    decoder_params: dict,
    selected_indices: np.ndarray,
    selected_ranks: np.ndarray,
    selected_probs: np.ndarray,
    selected_error_patterns: list[np.ndarray],
) -> list[dict]:
    """
    Process a chunk of shots for correlation analysis (worker function).

    Parameters
    ----------
    chunk_start : int
        Starting shot index for this chunk.
    chunk_size : int
        Number of shots in this chunk.
    det_chunk : 2D numpy array of bool
        Detector outcomes for this chunk, shape (chunk_size, n_detectors).
    circuit : stim.Circuit
        The circuit (used to create decoder).
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    selected_indices : 1D numpy array of int
        Selected error indices.
    selected_ranks : 1D numpy array of int
        Selected error ranks.
    selected_probs : 1D numpy array of float
        Selected error probabilities.
    selected_error_patterns : list of 1D numpy array of bool
        Selected error bit patterns.

    Returns
    -------
    results : list of dict
        Results for all shots in this chunk.
    """
    # Create decoder for this worker (decoders are not thread-safe)
    decoder = SoftOutputsBpLsdDecoder(circuit=circuit, **decoder_params)
    obs_matrix_T = decoder.obs_matrix.T

    results = []
    for i in range(chunk_size):
        shot_idx = chunk_start + i
        det = det_chunk[i]

        # Standard decoding
        pred, _, _, soft_info = decoder.decode(
            det,
            include_cluster_stats=False,
            compute_logical_gap_proxy=False,
        )

        # Get best logical class and LLR from standard decoding
        best_logical_class = ((pred.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)
        best_llr = soft_info["pred_llr"]

        # Fixed-class decoding for each selected error pattern
        for error_idx, error_rank, error_prob, error_pattern in zip(
            selected_indices, selected_ranks, selected_probs, selected_error_patterns
        ):
            # Candidate class = best_class XOR error_pattern
            candidate_class = best_logical_class ^ error_pattern

            # Fixed-class decoding
            fixed_llr, _ = decoder._perform_fixed_logical_class_decoding(
                det, candidate_class
            )

            # Record result
            results.append(
                {
                    "shot_id": shot_idx,
                    "error_rank": int(error_rank),
                    "error_index": int(error_idx),
                    "error_prob": float(error_prob),
                    "fixed_llr": float(fixed_llr),
                    "best_llr": float(best_llr),
                    "llr_delta": float(fixed_llr - best_llr),
                }
            )

    return results


def run_correlation_analysis(
    n_qubits: int,
    p: float,
    n_shots: int,
    n_representative_errors: int,
    decoder_params: dict,
    distribution_path: Path | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run correlation analysis between logical error distribution and fixed-class LLRs.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits (determines BB code variant).
    p : float
        Physical error rate.
    n_shots : int
        Number of shots to analyze.
    n_representative_errors : int
        Number of representative errors to sample from distribution.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    distribution_path : Path, optional
        Path to pre-computed distribution. If None, computes fresh.
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs. Use -1 for all CPUs.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: shot_id, error_rank, error_index, error_prob,
        fixed_llr, best_llr, llr_delta
    metadata : dict
        Metadata about the analysis run.
    """
    # Get code distance from n_qubits
    distance_map = {72: 6, 90: 10, 108: 10, 144: 12, 288: 18, 360: 24, 756: 34}
    if n_qubits not in distance_map:
        raise ValueError(
            f"Unsupported n_qubits: {n_qubits}. Must be one of {list(distance_map.keys())}"
        )
    distance = distance_map[n_qubits]
    T = distance  # measurement rounds = distance

    if verbose:
        print(f"Building [[{n_qubits}, 12, {distance}]] BB circuit with p={p}...")
    circuit = build_BB_circuit(n=n_qubits, T=T, p=p)
    num_observables = circuit.num_observables

    if verbose:
        print(
            f"Circuit: {circuit.num_detectors} detectors, {num_observables} observables"
        )

    # Load or compute distribution
    if distribution_path and distribution_path.exists():
        if verbose:
            print(f"Loading distribution from {distribution_path}...")
        distribution = np.load(distribution_path)
    else:
        if verbose:
            print("Computing fresh distribution (100k shots)...")
        distribution, _ = collect_logical_error_distribution_fast(
            circuit=circuit,
            shots=100_000,
            decoder_params=decoder_params,
            seed=seed,
        )
        if distribution_path:
            distribution_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(distribution_path, distribution)
            if verbose:
                print(f"Distribution saved to {distribution_path}")

    # Distribution statistics
    total_shots_dist = distribution.sum()
    logical_error_rate = 1 - distribution[0] / total_shots_dist

    if verbose:
        print(
            f"Distribution: {total_shots_dist:,} shots, {logical_error_rate:.4%} logical error rate"
        )

    # Select representative errors
    selected_indices, selected_ranks, selected_probs = select_representative_errors(
        distribution, n_samples=n_representative_errors
    )

    if verbose:
        print(f"\nSelected {n_representative_errors} representative errors:")
        print(f"  Ranks: {selected_ranks.tolist()}")
        print(f"  Indices: {selected_indices.tolist()}")

    # Convert selected error indices to bit patterns
    selected_error_patterns = [
        index_to_logical_class(int(idx), num_observables) for idx in selected_indices
    ]

    # Prepare sampler
    sampler = circuit.compile_detector_sampler(seed=seed + 1000)

    # Sample all detector outcomes at once
    if verbose:
        print(f"\nSampling {n_shots:,} detector outcomes...")
    det_all, _ = sampler.sample(n_shots, separate_observables=True)

    # Determine actual number of jobs
    if n_jobs == -1:
        import os

        actual_n_jobs = os.cpu_count() or 1
    else:
        actual_n_jobs = max(1, n_jobs)

    # Run analysis
    if actual_n_jobs == 1:
        # Sequential processing with progress bar
        if verbose:
            print(
                f"Running analysis sequentially ({n_shots:,} shots x {n_representative_errors + 1} decodes each)..."
            )

        # Create decoder
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit, **decoder_params)
        obs_matrix_T = decoder.obs_matrix.T

        results = []
        iterator = (
            tqdm(range(n_shots), desc="Processing shots") if verbose else range(n_shots)
        )

        for shot_idx in iterator:
            det = det_all[shot_idx]

            # Standard decoding
            pred, _, _, soft_info = decoder.decode(
                det,
                include_cluster_stats=False,
                compute_logical_gap_proxy=False,
            )

            # Get best logical class and LLR from standard decoding
            best_logical_class = ((pred.astype(np.uint8) @ obs_matrix_T) % 2).astype(
                bool
            )
            best_llr = soft_info["pred_llr"]

            # Fixed-class decoding for each selected error pattern
            for error_idx, error_rank, error_prob, error_pattern in zip(
                selected_indices,
                selected_ranks,
                selected_probs,
                selected_error_patterns,
            ):
                # Candidate class = best_class XOR error_pattern
                candidate_class = best_logical_class ^ error_pattern

                # Fixed-class decoding
                fixed_llr, _ = decoder._perform_fixed_logical_class_decoding(
                    det, candidate_class
                )

                # Record result
                results.append(
                    {
                        "shot_id": shot_idx,
                        "error_rank": int(error_rank),
                        "error_index": int(error_idx),
                        "error_prob": float(error_prob),
                        "fixed_llr": float(fixed_llr),
                        "best_llr": float(best_llr),
                        "llr_delta": float(fixed_llr - best_llr),
                    }
                )
    else:
        # Parallel processing
        if verbose:
            print(
                f"Running analysis in parallel ({actual_n_jobs} jobs, {n_shots:,} shots x {n_representative_errors + 1} decodes each)..."
            )

        # Calculate chunk sizes for each job
        base_chunk_size = n_shots // actual_n_jobs
        remainder = n_shots % actual_n_jobs

        chunks = []
        start = 0
        for i in range(actual_n_jobs):
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            if chunk_size > 0:
                chunks.append((start, chunk_size, det_all[start : start + chunk_size]))
            start += chunk_size

        # Process chunks in parallel
        results_chunks = Parallel(n_jobs=actual_n_jobs, verbose=10 if verbose else 0)(
            delayed(_process_shots_chunk)(
                chunk_start,
                chunk_size,
                det_chunk,
                circuit,
                decoder_params,
                selected_indices,
                selected_ranks,
                selected_probs,
                selected_error_patterns,
            )
            for chunk_start, chunk_size, det_chunk in chunks
        )

        # Flatten results
        results = [item for chunk_results in results_chunks for item in chunk_results]

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Metadata
    metadata = {
        "n_qubits": n_qubits,
        "distance": distance,
        "p": p,
        "n_shots": n_shots,
        "n_representative_errors": n_representative_errors,
        "num_observables": num_observables,
        "logical_error_rate": logical_error_rate,
        "distribution_shots": int(total_shots_dist),
        "decoder_params": decoder_params,
        "seed": seed,
        "n_jobs": actual_n_jobs,
        "selected_error_indices": selected_indices.tolist(),
        "selected_error_ranks": selected_ranks.tolist(),
        "selected_error_probs": selected_probs.tolist(),
    }

    return results_df, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between logical error distribution and fixed-class decoding LLRs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for results (parquet file)",
    )

    # Code parameters
    parser.add_argument(
        "--n-qubits",
        "-n",
        type=int,
        default=144,
        choices=[72, 90, 108, 144, 288, 360, 756],
        help="Number of physical qubits (BB code variant)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.003,
        help="Physical error rate",
    )

    # Analysis parameters
    parser.add_argument(
        "--n-shots",
        type=int,
        default=10000,
        help="Number of shots to analyze",
    )
    parser.add_argument(
        "--n-errors",
        type=int,
        default=10,
        help="Number of representative errors to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 for all CPUs)",
    )

    # Distribution path
    parser.add_argument(
        "--distribution-path",
        type=str,
        default=None,
        help="Path to pre-computed distribution (optional)",
    )

    # Decoder parameters
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Maximum BP iterations",
    )
    parser.add_argument(
        "--bp-method",
        type=str,
        default="minimum_sum",
        help="BP method",
    )
    parser.add_argument(
        "--lsd-method",
        type=str,
        default="LSD_0",
        help="LSD method",
    )
    parser.add_argument(
        "--lsd-order",
        type=int,
        default=0,
        help="LSD order",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Build decoder params
    decoder_params = {
        "max_iter": args.max_iter,
        "bp_method": args.bp_method,
        "lsd_method": args.lsd_method,
        "lsd_order": args.lsd_order,
    }

    # Distribution path
    dist_path = Path(args.distribution_path) if args.distribution_path else None

    # If no distribution path provided, use default location
    if dist_path is None:
        distance_map = {72: 6, 90: 10, 108: 10, 144: 12, 288: 18, 360: 24, 756: 34}
        distance = distance_map[args.n_qubits]
        T = distance
        bp_method_short = (
            "minsum" if args.bp_method == "minimum_sum" else args.bp_method
        )
        lsd_method_short = args.lsd_method.lower().replace("_", "")
        dist_dir = (
            project_root
            / "simulations"
            / "data"
            / "logical_error_distributions"
            / f"bb_{bp_method_short}_iter{args.max_iter}_{lsd_method_short}"
        )
        dist_path = dist_dir / f"n{args.n_qubits}_T{T}_p{args.p}.npy"

    # Run analysis
    results_df, metadata = run_correlation_analysis(
        n_qubits=args.n_qubits,
        p=args.p,
        n_shots=args.n_shots,
        n_representative_errors=args.n_errors,
        decoder_params=decoder_params,
        distribution_path=dist_path,
        seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame to parquet
    results_df.to_parquet(output_path, index=False)

    # Save metadata to JSON alongside
    metadata_path = output_path.with_suffix(".json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to: {output_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Total records: {len(results_df):,}")


if __name__ == "__main__":
    main()
