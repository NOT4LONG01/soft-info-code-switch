"""
Module for collecting logical error distributions from decoding simulations.

This module provides utilities to run basic decoding simulations and collect
statistics on the distribution of logical errors. The resulting distribution
can be used with the 'most-likely-first' gap proxy method.
"""

from typing import Any, Dict, Tuple

import numpy as np
import stim


def logical_class_to_index(logical_class: np.ndarray) -> int:
    """
    Convert a logical class/error bit pattern to an integer index.

    The index is computed as i = sum(b_j * 2^j) for j=0..k-1,
    where b_j is the j-th bit of the logical class.

    Parameters
    ----------
    logical_class : 1D numpy array of bool
        Bit pattern of the logical class/error.

    Returns
    -------
    index : int
        Integer index representing the logical class/error.

    Examples
    --------
    >>> logical_class_to_index(np.array([False, False, False]))
    0
    >>> logical_class_to_index(np.array([True, False, False]))
    1
    >>> logical_class_to_index(np.array([False, True, False]))
    2
    >>> logical_class_to_index(np.array([True, True, False]))
    3
    """
    num_observables = len(logical_class)
    if num_observables == 0:
        return 0

    if num_observables <= 64:
        # Vectorized path for typical cases
        powers_of_two = np.uint64(1) << np.arange(num_observables, dtype=np.uint64)
        return int(logical_class.astype(np.uint64) @ powers_of_two)
    else:
        # Fallback for >64 observables (uses Python arbitrary-precision int)
        index = 0
        for bit_idx, bit in enumerate(logical_class.tolist()):
            if bit:
                index |= 1 << bit_idx
        return index


def index_to_logical_class(index: int, num_observables: int) -> np.ndarray:
    """
    Convert an integer index to a logical class/error bit pattern.

    The bit pattern is determined by i = sum(b_j * 2^j) for j=0..k-1,
    where b_j is the j-th bit of the logical class.

    Parameters
    ----------
    index : int
        Integer index representing the logical class/error.
    num_observables : int
        Number of observables (k).

    Returns
    -------
    logical_class : 1D numpy array of bool with shape (num_observables,)
        Bit pattern where logical_class[j] = (index >> j) & 1.

    Examples
    --------
    >>> index_to_logical_class(0, 3)
    array([False, False, False])
    >>> index_to_logical_class(1, 3)
    array([ True, False, False])
    >>> index_to_logical_class(5, 3)
    array([ True, False,  True])
    """
    if num_observables <= 64:
        # Vectorized path for typical cases
        bit_positions = np.arange(num_observables, dtype=np.uint64)
        logical_class = ((np.uint64(index) >> bit_positions) & 1).astype(bool)
    else:
        # Fallback for >64 observables
        logical_class = np.array(
            [(index >> j) & 1 for j in range(num_observables)],
            dtype=bool,
        )
    return logical_class


def normalize_distribution(distribution: np.ndarray) -> np.ndarray:
    """
    Normalize a distribution array to sum to 1 (convert counts to probabilities).

    Parameters
    ----------
    distribution : 1D numpy array of float or int
        Array of counts or unnormalized probabilities.

    Returns
    -------
    normalized : 1D numpy array of float
        Normalized probability distribution that sums to 1.
        Returns uniform distribution if input sums to 0.
    """
    total = distribution.sum()
    if total == 0:
        return np.ones_like(distribution, dtype=float) / len(distribution)
    return distribution.astype(float) / total


def collect_logical_error_distribution(
    circuit: stim.Circuit,
    shots: int,
    decoder: "SoftOutputsBpLsdDecoder | None" = None,
    decoder_params: Dict[str, Any] | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Collect logical error distribution by running basic decoding simulation.

    Runs BP+LSD decoding on sampled detector outcomes and collects statistics
    on the distribution of logical errors (XOR of true and predicted logical
    classes).

    Parameters
    ----------
    circuit : stim.Circuit
        Quantum error correction circuit to simulate.
    shots : int
        Number of simulation shots.
    decoder : SoftOutputsBpLsdDecoder, optional
        Pre-initialized decoder. If None, creates one from circuit using
        decoder_params. Defaults to None.
    decoder_params : dict, optional
        Parameters for decoder initialization (if decoder is None).
        Passed as keyword arguments to SoftOutputsBpLsdDecoder constructor.
        Defaults to None (empty dict).
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    batch_size : int, optional
        Number of shots to process in each batch. If None, processes all shots
        at once. Useful for memory management with large shot counts.
        Defaults to None.

    Returns
    -------
    distribution : 1D numpy array of int with shape (2^k,)
        Counts of each logical error where k is the number of observables.
        Index i corresponds to logical error with bit pattern i = sum(b_j * 2^j).
        Index 0 represents no logical error (correct decoding).
    metadata : dict
        Dictionary containing simulation metadata:
        - 'total_shots': Total number of simulation shots.
        - 'num_observables': Number of observables (k).
        - 'logical_error_rate': Overall logical error rate (1 - distribution[0] / total_shots).
        - 'nonzero_errors': Number of distinct nonzero logical errors observed.

    Examples
    --------
    >>> import stim
    >>> circuit = stim.Circuit.generated(
    ...     "repetition_code:memory",
    ...     distance=3,
    ...     rounds=3,
    ...     after_clifford_depolarization=0.01,
    ... )
    >>> distribution, metadata = collect_logical_error_distribution(circuit, shots=1000)
    >>> print(f"Logical error rate: {metadata['logical_error_rate']:.4f}")
    """
    # Import here to avoid circular import
    from ..decoders.bplsd import SoftOutputsBpLsdDecoder

    if decoder is None:
        if decoder_params is None:
            decoder_params = {}
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit, **decoder_params)

    # Get number of observables
    num_observables = (
        decoder.obs_matrix.shape[0] if decoder.obs_matrix is not None else 0
    )
    if num_observables == 0:
        raise ValueError(
            "Circuit has no observables; cannot compute logical error distribution."
        )

    # Initialize distribution array
    total_num_logical_classes = 1 << num_observables
    distribution = np.zeros(total_num_logical_classes, dtype=np.int64)

    # Create sampler with seed
    sampler = circuit.compile_detector_sampler(seed=seed)

    # Determine batch processing
    if batch_size is None or batch_size >= shots:
        batch_sizes = [shots]
    else:
        num_full_batches = shots // batch_size
        remainder = shots % batch_size
        batch_sizes = [batch_size] * num_full_batches
        if remainder > 0:
            batch_sizes.append(remainder)

    # Process batches
    obs_matrix_T = decoder.obs_matrix.T
 
    for current_batch_size in batch_sizes:
        # Sample detector outcomes and true observables
        det, obs = sampler.sample(current_batch_size, separate_observables=True)

        # Decode all shots in this batch
        for i in range(current_batch_size):
            # Decode without gap proxy computation for efficiency
            pred, _, _, _ = decoder.decode(
                det[i],
                include_cluster_stats=False,
                compute_logical_gap_proxy=False,
            )

            # Compute predicted logical class
            pred_obs = ((pred.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)

            # Compute logical error (XOR of true and predicted)
            logical_error = obs[i] ^ pred_obs

            # Convert to index and increment count
            error_idx = logical_class_to_index(logical_error)
            distribution[error_idx] += 1

    # Compute metadata
    total_shots_processed = sum(batch_sizes)
    correct_count = distribution[0]
    logical_error_rate = 1.0 - (correct_count / total_shots_processed)
    nonzero_errors = np.sum(distribution[1:] > 0)

    metadata = {
        "total_shots": total_shots_processed,
        "num_observables": num_observables,
        "logical_error_rate": logical_error_rate,
        "nonzero_errors": int(nonzero_errors),
    }

    return distribution, metadata


def collect_logical_error_distribution_fast(
    circuit: stim.Circuit,
    shots: int,
    decoder: "SoftOutputsBpLsdDecoder | None" = None,
    decoder_params: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Collect logical error distribution using vectorized operations for speed.

    This is a faster version of collect_logical_error_distribution that
    decodes all shots first and then computes error statistics in bulk.
    Suitable when memory is not a constraint.

    Parameters
    ----------
    circuit : stim.Circuit
        Quantum error correction circuit to simulate.
    shots : int
        Number of simulation shots.
    decoder : SoftOutputsBpLsdDecoder, optional
        Pre-initialized decoder. If None, creates one from circuit using
        decoder_params. Defaults to None.
    decoder_params : dict, optional
        Parameters for decoder initialization (if decoder is None).
        Passed as keyword arguments to SoftOutputsBpLsdDecoder constructor.
        Defaults to None (empty dict).
    seed : int, optional
        Random seed for reproducibility. Defaults to None.

    Returns
    -------
    distribution : 1D numpy array of int with shape (2^k,)
        Counts of each logical error where k is the number of observables.
    metadata : dict
        Dictionary containing simulation metadata.
    """
    # Import here to avoid circular import
    from ..decoders.bplsd import SoftOutputsBpLsdDecoder

    if decoder is None:
        if decoder_params is None:
            decoder_params = {}
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit, **decoder_params)

    # Get number of observables
    num_observables = (
        decoder.obs_matrix.shape[0] if decoder.obs_matrix is not None else 0
    )
    if num_observables == 0:
        raise ValueError(
            "Circuit has no observables; cannot compute logical error distribution."
        )

    # Initialize distribution array
    total_num_logical_classes = 1 << num_observables
    distribution = np.zeros(total_num_logical_classes, dtype=np.int64)

    # Create sampler with seed
    sampler = circuit.compile_detector_sampler(seed=seed)

    # Sample all detector outcomes and true observables at once
    det, obs = sampler.sample(shots, separate_observables=True)

    # Decode all shots and collect predictions
    preds_list = []
    for i in range(shots):
        pred, _, _, _ = decoder.decode(
            det[i],
            include_cluster_stats=False,
            compute_logical_gap_proxy=False,
        )
        preds_list.append(pred)

    preds_arr = np.array(preds_list)

    # Compute predicted logical classes in bulk
    obs_matrix_T = decoder.obs_matrix.T
    pred_obs_arr = ((preds_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)

    # Compute logical errors in bulk
    logical_errors = obs ^ pred_obs_arr  # Shape: (shots, num_observables)

    # Convert each logical error to index and count
    if num_observables <= 64:
        # Vectorized conversion to indices
        powers_of_two = np.uint64(1) << np.arange(num_observables, dtype=np.uint64)
        error_indices = logical_errors.astype(np.uint64) @ powers_of_two
        # Count occurrences
        unique, counts = np.unique(error_indices, return_counts=True)
        distribution[unique.astype(int)] = counts
    else:
        # Fallback for >64 observables
        for i in range(shots):
            error_idx = logical_class_to_index(logical_errors[i])
            distribution[error_idx] += 1

    # Compute metadata
    correct_count = distribution[0]
    logical_error_rate = 1.0 - (correct_count / shots)
    nonzero_errors = np.sum(distribution[1:] > 0)

    metadata = {
        "total_shots": shots,
        "num_observables": num_observables,
        "logical_error_rate": logical_error_rate,
        "nonzero_errors": int(nonzero_errors),
    }

    return distribution, metadata
