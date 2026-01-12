#!/usr/bin/env python
"""
Script to analyze order-dependent behavior in ldpc BpLsdDecoder.

This script tests for a known upstream bug where uninitialized variables
can cause undefined behavior on all-zero syndromes and state leakage
between decode() calls.

Bug details:
- In ldpc's _bplsd_decoder.pyx, `zero_syndrome` is not initialized before use
- With Cython's initializedcheck=False, this causes undefined behavior
- Internal state (bp_decoding, statistics) may leak between calls

Tests:
1. All-zero syndrome consistency: Repeated decodes should give identical results
2. State leakage: bp_decoding should be properly reset after zero-syndrome
3. Order dependence: Same syndrome should give same result regardless of decode order
4. Fixed-class decoder: Cached decoder should not introduce order dependence

Usage:
    python run_ldpc_order_dependence_analysis.py --help
    python run_ldpc_order_dependence_analysis.py --output results.parquet
    python run_ldpc_order_dependence_analysis.py --n-qubits 72 --n-shots 100 --n-repeats 20
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "simulations"))

from ldpc_post_selection.bplsd_decoder import SoftOutputsBpLsdDecoder
from simulations.utils.build_circuit import build_BB_circuit


def _array_hash(arr: np.ndarray) -> str:
    """Compute a hash of a numpy array for quick comparison."""
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _dict_hash(d: dict) -> str:
    """Compute a hash of a dictionary for quick comparison."""
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _create_decoder(circuit, decoder_params: dict) -> SoftOutputsBpLsdDecoder:
    """Create a fresh decoder instance."""
    return SoftOutputsBpLsdDecoder(circuit=circuit, **decoder_params)


def _get_internal_state(decoder: SoftOutputsBpLsdDecoder) -> dict:
    """Extract internal state from underlying BpLsdDecoder."""
    bplsd = decoder._bplsd
    return {
        "converge": bool(bplsd.converge),
        "bp_decoding": bplsd.bp_decoding.copy(),
        "statistics": dict(bplsd.statistics),
    }


def run_test1_zero_syndrome_consistency(
    circuit,
    decoder_params: dict,
    n_repeats: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Test 1: All-Zero Syndrome Consistency.

    Decode all-zero syndrome N times with same decoder instance.
    Check if results are identical across all calls.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit for decoding.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    n_repeats : int
        Number of repeated decodes.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : list of dict
        Results for each iteration.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1: All-Zero Syndrome Consistency")
        print("=" * 60)

    decoder = _create_decoder(circuit, decoder_params)
    num_detectors = decoder.H.shape[0]
    zero_syndrome = np.zeros(num_detectors, dtype=bool)

    results = []
    first_result = None

    iterator = tqdm(range(n_repeats), desc="Test 1") if verbose else range(n_repeats)
    for i in iterator:
        pred, pred_bp, converge, soft_outputs = decoder.decode(
            zero_syndrome, include_cluster_stats=True
        )
        state = _get_internal_state(decoder)

        result = {
            "test_name": "zero_syndrome_consistency",
            "iteration": i,
            "pred_hash": _array_hash(pred),
            "pred_bp_hash": _array_hash(pred_bp),
            "pred_llr": float(soft_outputs["pred_llr"]),
            "converge": converge,
            "bp_decoding_hash": _array_hash(state["bp_decoding"]),
            "statistics_hash": _dict_hash(state["statistics"]),
            "pred_sum": int(pred.sum()),
            "pred_bp_sum": int(pred_bp.sum()),
        }

        if first_result is None:
            first_result = result.copy()
            result["match_first"] = True
        else:
            result["match_first"] = (
                result["pred_hash"] == first_result["pred_hash"]
                and result["pred_bp_hash"] == first_result["pred_bp_hash"]
                and np.isclose(result["pred_llr"], first_result["pred_llr"])
                and result["converge"] == first_result["converge"]
                and result["bp_decoding_hash"] == first_result["bp_decoding_hash"]
            )

        results.append(result)

    if verbose:
        n_matches = sum(1 for r in results if r["match_first"])
        print(f"Consistency: {n_matches}/{n_repeats} iterations match first result")
        if n_matches < n_repeats:
            print("WARNING: Inconsistent results detected!")

    return results


def run_test2_state_leakage(
    circuit,
    decoder_params: dict,
    n_shots: int,
    seed: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Test 2: State Leakage After Non-Trivial Decode.

    For each shot:
    1. Decode a non-trivial syndrome (from sampling)
    2. Decode all-zero syndrome
    3. Check if bp_decoding is properly reset

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit for decoding.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    n_shots : int
        Number of syndrome samples to test.
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : list of dict
        Results for each shot.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 2: State Leakage After Non-Trivial Decode")
        print("=" * 60)

    # Sample syndromes
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, _ = sampler.sample(shots=n_shots, separate_observables=True)

    decoder = _create_decoder(circuit, decoder_params)
    num_detectors = decoder.H.shape[0]
    zero_syndrome = np.zeros(num_detectors, dtype=bool)

    results = []
    iterator = tqdm(range(n_shots), desc="Test 2") if verbose else range(n_shots)

    for shot_idx in iterator:
        nontrivial_syndrome = detection_events[shot_idx]

        # Skip if syndrome is already all zeros
        if nontrivial_syndrome.sum() == 0:
            continue

        # Decode non-trivial syndrome first
        pred_nt, pred_bp_nt, converge_nt, soft_nt = decoder.decode(
            nontrivial_syndrome, include_cluster_stats=True
        )
        state_after_nontrivial = _get_internal_state(decoder)

        # Now decode all-zero syndrome
        pred_zero, pred_bp_zero, converge_zero, soft_zero = decoder.decode(
            zero_syndrome, include_cluster_stats=True
        )
        state_after_zero = _get_internal_state(decoder)

        # Check for state leakage
        bp_decoding_leaked = np.array_equal(
            state_after_zero["bp_decoding"], state_after_nontrivial["bp_decoding"]
        )
        statistics_leaked = _dict_hash(state_after_zero["statistics"]) == _dict_hash(
            state_after_nontrivial["statistics"]
        )

        result = {
            "test_name": "state_leakage",
            "shot_id": shot_idx,
            "nontrivial_violations": int(nontrivial_syndrome.sum()),
            "nontrivial_pred_sum": int(pred_nt.sum()),
            "nontrivial_pred_llr": float(soft_nt["pred_llr"]),
            "zero_pred_sum": int(pred_zero.sum()),
            "zero_pred_llr": float(soft_zero["pred_llr"]),
            "zero_pred_bp_sum": int(pred_bp_zero.sum()),
            "bp_decoding_leaked": bp_decoding_leaked,
            "statistics_leaked": statistics_leaked,
            "bp_decoding_hash_nt": _array_hash(state_after_nontrivial["bp_decoding"]),
            "bp_decoding_hash_zero": _array_hash(state_after_zero["bp_decoding"]),
        }
        results.append(result)

    if verbose:
        n_bp_leaked = sum(1 for r in results if r["bp_decoding_leaked"])
        n_stats_leaked = sum(1 for r in results if r["statistics_leaked"])
        print(f"bp_decoding leaked: {n_bp_leaked}/{len(results)} shots")
        print(f"statistics leaked: {n_stats_leaked}/{len(results)} shots")
        if n_bp_leaked > 0 or n_stats_leaked > 0:
            print("WARNING: State leakage detected!")

    return results


def run_test3_order_dependence(
    circuit,
    decoder_params: dict,
    n_shots: int,
    seed: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Test 3: Order Dependence (Two Decoders).

    For pairs of syndromes (A, B):
    - Decoder 1: decode(A), decode(B)
    - Decoder 2: decode(B), decode(A)
    Check if results for same syndrome match regardless of decode order.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit for decoding.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    n_shots : int
        Number of syndrome pairs to test.
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : list of dict
        Results for each syndrome pair.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 3: Order Dependence (Two Decoders)")
        print("=" * 60)

    # Sample syndromes (need 2 * n_shots for pairs)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, _ = sampler.sample(shots=2 * n_shots, separate_observables=True)

    results = []
    iterator = tqdm(range(n_shots), desc="Test 3") if verbose else range(n_shots)

    for pair_idx in iterator:
        syndrome_a = detection_events[2 * pair_idx]
        syndrome_b = detection_events[2 * pair_idx + 1]

        # Decoder 1: A then B
        decoder1 = _create_decoder(circuit, decoder_params)
        pred_a1, pred_bp_a1, conv_a1, soft_a1 = decoder1.decode(syndrome_a)
        pred_b1, pred_bp_b1, conv_b1, soft_b1 = decoder1.decode(syndrome_b)

        # Decoder 2: B then A
        decoder2 = _create_decoder(circuit, decoder_params)
        pred_b2, pred_bp_b2, conv_b2, soft_b2 = decoder2.decode(syndrome_b)
        pred_a2, pred_bp_a2, conv_a2, soft_a2 = decoder2.decode(syndrome_a)

        # Check if results match
        a_pred_match = np.array_equal(pred_a1, pred_a2)
        a_pred_bp_match = np.array_equal(pred_bp_a1, pred_bp_a2)
        a_llr_match = np.isclose(soft_a1["pred_llr"], soft_a2["pred_llr"])

        b_pred_match = np.array_equal(pred_b1, pred_b2)
        b_pred_bp_match = np.array_equal(pred_bp_b1, pred_bp_b2)
        b_llr_match = np.isclose(soft_b1["pred_llr"], soft_b2["pred_llr"])

        result = {
            "test_name": "order_dependence",
            "pair_id": pair_idx,
            "syndrome_a_violations": int(syndrome_a.sum()),
            "syndrome_b_violations": int(syndrome_b.sum()),
            # Syndrome A results
            "a_pred_match": a_pred_match,
            "a_pred_bp_match": a_pred_bp_match,
            "a_llr_match": a_llr_match,
            "a_llr_order1": float(soft_a1["pred_llr"]),
            "a_llr_order2": float(soft_a2["pred_llr"]),
            "a_llr_diff": float(soft_a1["pred_llr"] - soft_a2["pred_llr"]),
            # Syndrome B results
            "b_pred_match": b_pred_match,
            "b_pred_bp_match": b_pred_bp_match,
            "b_llr_match": b_llr_match,
            "b_llr_order1": float(soft_b1["pred_llr"]),
            "b_llr_order2": float(soft_b2["pred_llr"]),
            "b_llr_diff": float(soft_b1["pred_llr"] - soft_b2["pred_llr"]),
            # Overall
            "all_match": (
                a_pred_match
                and a_pred_bp_match
                and a_llr_match
                and b_pred_match
                and b_pred_bp_match
                and b_llr_match
            ),
        }
        results.append(result)

    if verbose:
        n_all_match = sum(1 for r in results if r["all_match"])
        n_a_mismatch = sum(1 for r in results if not r["a_pred_match"])
        n_b_mismatch = sum(1 for r in results if not r["b_pred_match"])
        print(f"All results match: {n_all_match}/{len(results)} pairs")
        print(f"Syndrome A mismatches: {n_a_mismatch}/{len(results)}")
        print(f"Syndrome B mismatches: {n_b_mismatch}/{len(results)}")
        if n_all_match < len(results):
            print("WARNING: Order-dependent behavior detected!")

    return results


def run_test4_fixed_class_order_dependence(
    circuit,
    decoder_params: dict,
    n_shots: int,
    seed: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Test 4: Fixed-Class Decoder Order Dependence.

    Test the cached _decoder_for_fixed_class used in gap proxy computation.
    For each shot, decode with logical class 0 then 1 vs 1 then 0.
    Check if pred_llr differs based on exploration order.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit for decoding.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    n_shots : int
        Number of shots to test.
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : list of dict
        Results for each shot.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 4: Fixed-Class Decoder Order Dependence")
        print("=" * 60)

    # Sample syndromes
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, _ = sampler.sample(shots=n_shots, separate_observables=True)

    # Create decoder to check obs_matrix
    test_decoder = _create_decoder(circuit, decoder_params)
    if test_decoder.obs_matrix is None:
        if verbose:
            print("SKIPPED: No obs_matrix available")
        return []

    num_obs = test_decoder.obs_matrix.shape[0]
    del test_decoder

    # Define two logical classes to test
    class_0 = np.zeros(num_obs, dtype=bool)  # All False
    class_1 = np.zeros(num_obs, dtype=bool)
    class_1[0] = True  # First bit True

    results = []
    iterator = tqdm(range(n_shots), desc="Test 4") if verbose else range(n_shots)

    for shot_idx in iterator:
        syndrome = detection_events[shot_idx]

        # Order 1: class_0 then class_1 (fresh decoder)
        decoder1 = _create_decoder(circuit, decoder_params)
        fixed_decoder1 = decoder1._decoder_for_fixed_class

        syndrome_0_o1 = np.concatenate([syndrome, class_0])
        syndrome_1_o1 = np.concatenate([syndrome, class_1])

        _, _, _, soft_0_o1 = fixed_decoder1.decode(syndrome_0_o1)
        _, _, _, soft_1_o1 = fixed_decoder1.decode(syndrome_1_o1)

        # Order 2: class_1 then class_0 (fresh decoder)
        decoder2 = _create_decoder(circuit, decoder_params)
        fixed_decoder2 = decoder2._decoder_for_fixed_class

        syndrome_1_o2 = np.concatenate([syndrome, class_1])
        syndrome_0_o2 = np.concatenate([syndrome, class_0])

        _, _, _, soft_1_o2 = fixed_decoder2.decode(syndrome_1_o2)
        _, _, _, soft_0_o2 = fixed_decoder2.decode(syndrome_0_o2)

        # Check if results match
        class0_llr_match = np.isclose(soft_0_o1["pred_llr"], soft_0_o2["pred_llr"])
        class1_llr_match = np.isclose(soft_1_o1["pred_llr"], soft_1_o2["pred_llr"])

        result = {
            "test_name": "fixed_class_order",
            "shot_id": shot_idx,
            "syndrome_violations": int(syndrome.sum()),
            # Class 0 results
            "class0_llr_order1": float(soft_0_o1["pred_llr"]),
            "class0_llr_order2": float(soft_0_o2["pred_llr"]),
            "class0_llr_match": class0_llr_match,
            "class0_llr_diff": float(soft_0_o1["pred_llr"] - soft_0_o2["pred_llr"]),
            # Class 1 results
            "class1_llr_order1": float(soft_1_o1["pred_llr"]),
            "class1_llr_order2": float(soft_1_o2["pred_llr"]),
            "class1_llr_match": class1_llr_match,
            "class1_llr_diff": float(soft_1_o1["pred_llr"] - soft_1_o2["pred_llr"]),
            # Overall
            "all_match": class0_llr_match and class1_llr_match,
        }
        results.append(result)

    if verbose:
        n_all_match = sum(1 for r in results if r["all_match"])
        n_class0_mismatch = sum(1 for r in results if not r["class0_llr_match"])
        n_class1_mismatch = sum(1 for r in results if not r["class1_llr_match"])
        print(f"All results match: {n_all_match}/{len(results)} shots")
        print(f"Class 0 LLR mismatches: {n_class0_mismatch}/{len(results)}")
        print(f"Class 1 LLR mismatches: {n_class1_mismatch}/{len(results)}")
        if n_all_match < len(results):
            print("WARNING: Fixed-class decoder order dependence detected!")

    return results


def run_all_tests(
    n_qubits: int,
    p: float,
    n_shots: int,
    n_repeats: int,
    decoder_params: dict,
    seed: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all order dependence tests.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits (determines BB code variant).
    p : float
        Physical error rate.
    n_shots : int
        Number of shots for tests 2, 3, 4.
    n_repeats : int
        Number of repeats for test 1.
    decoder_params : dict
        Parameters for the BP+LSD decoder.
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results_df : pd.DataFrame
        Combined results from all tests.
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
    T = distance

    if verbose:
        print("=" * 60)
        print("LDPC ORDER DEPENDENCE ANALYSIS")
        print("=" * 60)
        print(f"Building [[{n_qubits}, 12, {distance}]] BB circuit with p={p}...")

    circuit = build_BB_circuit(n=n_qubits, T=T, p=p)

    if verbose:
        print(
            f"Circuit: {circuit.num_detectors} detectors, {circuit.num_observables} observables"
        )
        print(f"Decoder params: {decoder_params}")
        print(f"Seed: {seed}")

    # Run all tests
    all_results = []

    # Test 1: Zero syndrome consistency
    results1 = run_test1_zero_syndrome_consistency(
        circuit, decoder_params, n_repeats, verbose
    )
    all_results.extend(results1)

    # Test 2: State leakage
    results2 = run_test2_state_leakage(circuit, decoder_params, n_shots, seed, verbose)
    all_results.extend(results2)

    # Test 3: Order dependence
    results3 = run_test3_order_dependence(
        circuit, decoder_params, n_shots, seed + 1000, verbose
    )
    all_results.extend(results3)

    # Test 4: Fixed-class decoder
    results4 = run_test4_fixed_class_order_dependence(
        circuit, decoder_params, n_shots, seed + 2000, verbose
    )
    all_results.extend(results4)

    # Combine results
    results_df = pd.DataFrame(all_results)

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Test 1
        test1_pass = all(r.get("match_first", True) for r in results1)
        print(f"Test 1 (Zero syndrome consistency): {'PASS' if test1_pass else 'FAIL'}")

        # Test 2
        test2_pass = not any(r.get("bp_decoding_leaked", False) for r in results2)
        print(f"Test 2 (State leakage): {'PASS' if test2_pass else 'FAIL'}")

        # Test 3
        test3_pass = all(r.get("all_match", True) for r in results3)
        print(f"Test 3 (Order dependence): {'PASS' if test3_pass else 'FAIL'}")

        # Test 4
        test4_pass = all(r.get("all_match", True) for r in results4)
        print(f"Test 4 (Fixed-class order): {'PASS' if test4_pass else 'FAIL'}")

        all_pass = test1_pass and test2_pass and test3_pass and test4_pass
        print(f"\nOverall: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")

    # Metadata
    metadata = {
        "n_qubits": n_qubits,
        "distance": distance,
        "p": p,
        "n_shots": n_shots,
        "n_repeats": n_repeats,
        "decoder_params": decoder_params,
        "seed": seed,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "test1_count": len(results1),
        "test2_count": len(results2),
        "test3_count": len(results3),
        "test4_count": len(results4),
    }

    return results_df, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Analyze order-dependent behavior in ldpc BpLsdDecoder.",
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
        default=72,
        choices=[72, 90, 108, 144, 288, 360, 756],
        help="Number of physical qubits (BB code variant)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.01,
        help="Physical error rate",
    )

    # Test parameters
    parser.add_argument(
        "--n-shots",
        type=int,
        default=100,
        help="Number of shots for tests 2, 3, 4",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Number of repeats for test 1 (zero syndrome consistency)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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

    # Run all tests
    results_df, metadata = run_all_tests(
        n_qubits=args.n_qubits,
        p=args.p,
        n_shots=args.n_shots,
        n_repeats=args.n_repeats,
        decoder_params=decoder_params,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame to parquet
    results_df.to_parquet(output_path, index=False)

    # Save metadata to JSON alongside
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to: {output_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Total records: {len(results_df):,}")


if __name__ == "__main__":
    main()
