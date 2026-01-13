from itertools import product
import random
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Self

import numpy as np
import stim
from joblib import Parallel, delayed
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix, vstack

from .base import SoftOutputsDecoder
from .cluster_tools import compute_cluster_stats


def _decode_single_logical_class(
    H_obs_appended: csc_matrix,
    priors: np.ndarray,
    bplsd_kwargs: dict,
    detector_outcomes: np.ndarray,
    logical_class: np.ndarray,
) -> Tuple[Tuple[bool, ...], float, np.ndarray]:
    """
    Decode a single logical class in isolation for parallel execution.

    This function creates its own decoder instance to ensure thread-safety
    when called in parallel.

    Parameters
    ----------
    H_obs_appended : scipy csc_matrix of uint8
        Parity check matrix with observable matrix appended.
    priors : 1D numpy array of float
        Error probabilities.
    bplsd_kwargs : dict
        Keyword arguments for BpLsdDecoder initialization.
    detector_outcomes : 1D numpy array of bool
        Detector measurement outcomes.
    logical_class : 1D numpy array of bool
        Fixed logical class to decode with.

    Returns
    -------
    logical_class_tuple : tuple of bool
        The logical class as a tuple (for use as dict key).
    pred_llr : float
        Prediction LLR for the fixed logical class decoding.
    pred : 1D numpy array of bool
        Predicted error pattern.
    """
    # Import here to avoid circular import (function is at module level)
    decoder = SoftOutputsBpLsdDecoder(
        H=H_obs_appended,
        p=priors,
        obs_matrix=None,
        **bplsd_kwargs,
    )

    detector_outcomes_obs_appended = np.concatenate([detector_outcomes, logical_class])

    pred, _, _, soft_outputs = decoder.decode(
        detector_outcomes_obs_appended,
        include_cluster_stats=False,
        compute_logical_gap_proxy=False,
        verbose=False,
    )

    return tuple(logical_class), soft_outputs["pred_llr"], pred


def _compute_intermediate_gap_proxies_posthoc(
    original_pred_llr: float,
    results: List[Tuple[Tuple[bool, ...], float, np.ndarray]],
) -> Dict[int, float]:
    """
    Compute intermediate gap proxies from parallel execution results.

    Since parallel execution doesn't have a deterministic order that matches
    the sampling order, we compute intermediate gap proxies based on the
    order results were specified (which preserves the original sampling order
    when using joblib with ordered results).

    Parameters
    ----------
    original_pred_llr : float
        The prediction LLR from the initial (best) logical class.
    results : list of tuple
        List of (logical_class_tuple, pred_llr, pred_pattern) from parallel
        execution, in the original order they were submitted.

    Returns
    -------
    gap_proxies_by_num_classes : dict of int to float
        Dictionary mapping number of explored classes to gap proxy value.
    """
    gap_proxies_by_num_classes: Dict[int, float] = {}

    running_best_llr = float(original_pred_llr)
    running_second_best_llr = float("inf")
    explored_count = 1  # Initial class already counted

    for _, pred_llr, _ in results:
        explored_count += 1
        pred_llr_float = float(pred_llr)

        if pred_llr_float <= running_best_llr:
            running_second_best_llr = running_best_llr
            running_best_llr = pred_llr_float
        elif pred_llr_float < running_second_best_llr:
            running_second_best_llr = pred_llr_float

        if explored_count >= 2:
            effective_second = (
                running_second_best_llr
                if running_second_best_llr != float("inf")
                else running_best_llr
            )
            gap_proxies_by_num_classes[explored_count] = float(
                effective_second - running_best_llr
            )

    return gap_proxies_by_num_classes


class SoftOutputsBpLsdDecoder(SoftOutputsDecoder):
    """
    BP+LSD decoder with additional soft outputs for quantifying decoding confidence.
    """

    _bplsd: BpLsdDecoder

    def __init__(
        self,
        H: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        *,
        p: Optional[np.ndarray | List[float]] = None,
        obs_matrix: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        circuit: Optional[stim.Circuit] = None,
        max_iter: int = 30,
        bp_method: str = "product_sum",
        lsd_method: str = "LSD_0",
        lsd_order: int = 0,
        ms_scaling_factor: float = 1.0,
        detector_time_coords: int | Sequence[int] = -1,
        **kwargs,
    ):
        """
        BP+LSD decoder with additional soft outputs.

        Parameters
        ----------
        H : 2D array-like of bool/int, including scipy csc matrix
            Parity check matrix. Internally stored as a scipy csc matrix of uint8.
        p : 1D array-like of float
            Error probabilities.
        obs_matrix : 2D array-like of bool/int, including scipy csc matrix
            Observable matrix. Internally stored as a scipy csc matrix of uint8.
        circuit : stim.Circuit, optional
            Circuit.
        max_iter : int
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to
            "product_sum".
        lsd_method : str
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
        lsd_order : int
            Order parameter for LSD. Defaults to 0.
        ms_scaling_factor : float
            Scaling factor for min-sum BP. Defaults to 1.0.
        detector_time_coords : int or sequence of int, defaults to -1
            Time coordinates of the detectors for sliding window decoding.
            If not given, the last element of coordinates is used for each detector.
            If a single integer, it indicates which element of the coordinates to use.
            If a sequence of integers, it explicitly specifies the time coordinates of
            the detectors, so its length must be the same as the number of detectors.
            If `circuit` is not given, this must be a sequence of integers.
        """
        # SoftOutputsBpLsdDecoder will always use decompose_errors=False if a circuit is given
        super().__init__(
            H=H, p=p, obs_matrix=obs_matrix, circuit=circuit, decompose_errors=False
        )

        bplsd_kwargs = {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "lsd_method": lsd_method,
            "lsd_order": lsd_order,
            "ms_scaling_factor": ms_scaling_factor,
        }
        bplsd_kwargs.update(kwargs)

        self._bplsd_kwargs = bplsd_kwargs

        self._bplsd = BpLsdDecoder(
            self.H,
            error_channel=self.priors,
            always_run_lsd=True,
            **bplsd_kwargs,
        )
        self._bplsd.set_do_stats(True)

        try:
            if len(detector_time_coords) == self.H.shape[0]:
                self._detector_time_coords = np.array(detector_time_coords, dtype=int)
                self._det_time_coord_index = None
            else:
                raise ValueError(
                    "detector_time_coords must be a sequence of integers with the same length as the number of detectors"
                )
        except TypeError:
            self._detector_time_coords = None
            self._det_time_coord_index = detector_time_coords

        # Initialize caches for sliding window decoding
        self._window_structure_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._decoder_cache: Dict[str, SoftOutputsBpLsdDecoder] = {}

        # Cache for coverage-restricted eligible error indices
        # Key: (distribution_id, coverage_fraction, num_observables)
        # Value: eligible_error_indices array
        self._coverage_eligible_cache: Dict[Tuple[int, float, int], np.ndarray] = {}

        # Precompute adjacency matrix for efficient cluster labeling
        self._adjacency_matrix = (self.H.T @ self.H == 1).astype(bool)

    @property
    def detector_time_coords(self) -> np.ndarray:
        if self._detector_time_coords is not None:
            return self._detector_time_coords.copy()

        else:
            if self.circuit is None:
                raise ValueError(
                    "detector_time_coords must be a sequence of integers if circuit is not given"
                )

            det_coords_dict = self.circuit.get_detector_coordinates()
            det_indices = sorted(det_coords_dict.keys())
            det_time_coords = [
                det_coords_dict[i][self._det_time_coord_index] for i in det_indices
            ]
            det_time_coords = np.array(det_time_coords, dtype=int)
            self._detector_time_coords = det_time_coords
            return det_time_coords.copy()

    @property
    def _H_obs_appended(self) -> Optional[csc_matrix]:
        """Lazily compute and cache H with obs_matrix appended."""
        if not hasattr(self, "_H_obs_appended_cache"):
            if self.obs_matrix is None:
                self._H_obs_appended_cache = None
            else:
                self._H_obs_appended_cache = vstack(
                    [self.H, self.obs_matrix], format="csc", dtype="uint8"
                )
        return self._H_obs_appended_cache

    @property
    def _decoder_for_fixed_class(self) -> Optional["SoftOutputsBpLsdDecoder"]:
        """Lazily create and cache decoder for fixed-logical-class decoding."""
        if not hasattr(self, "_decoder_for_fixed_class_cache"):
            if self._H_obs_appended is None:
                self._decoder_for_fixed_class_cache = None
            else:
                self._decoder_for_fixed_class_cache = SoftOutputsBpLsdDecoder(
                    H=self._H_obs_appended,
                    p=self.priors,
                    obs_matrix=None,  # Prevent recursion
                    max_iter=self._bplsd.max_iter,
                    bp_method=self._bplsd.bp_method,
                    lsd_method=self._bplsd.lsd_method,
                    lsd_order=self._bplsd.lsd_order,
                    ms_scaling_factor=self._bplsd.ms_scaling_factor,
                )
        return self._decoder_for_fixed_class_cache

    def _check_detector_time_coords_validity(self):
        time_coords = self.detector_time_coords
        if min(time_coords) != 0:
            raise ValueError("Detector time coordinates must start from 0")

    def _get_logical_classes_to_explore(
        self,
        predicted_logical_class: np.ndarray,
        explore_only_nearby_logical_classes: bool,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Determine logical classes to explore for gap proxy computation.

        Parameters
        ----------
        predicted_logical_class : 1D numpy array of bool
            Predicted logical class.
        explore_only_nearby_logical_classes : bool
            If True, only explore adjacent logical classes (flip one bit).
            If False, explore all possible logical classes except the predicted one.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        logical_classes_to_explore : list of 1D numpy array of bool
            List of logical classes to explore.
        """
        num_observables = len(predicted_logical_class)
        logical_classes_to_explore = []

        if verbose:
            print(
                f"  Getting logical classes to explore (num_observables={num_observables})"
            )

        if explore_only_nearby_logical_classes:
            # Only adjacent logical classes (flip one bit at a time)
            for i in range(num_observables):
                nearby_logical_class = predicted_logical_class.copy()
                nearby_logical_class[i] = not nearby_logical_class[i]
                logical_classes_to_explore.append(nearby_logical_class)
            if verbose:
                print(
                    f"  Exploring {len(logical_classes_to_explore)} nearby logical classes"
                )
        else:
            # All possible logical classes except the predicted one
            all_logical_classes = product([False, True], repeat=num_observables)
            for logical_class in all_logical_classes:
                logical_class_array = np.array(logical_class, dtype=bool)
                if not np.array_equal(logical_class_array, predicted_logical_class):
                    logical_classes_to_explore.append(logical_class_array)
            if verbose:
                print(
                    f"  Exploring {len(logical_classes_to_explore)} total logical classes"
                )

        return logical_classes_to_explore

    def _sample_random_logical_classes(
        self,
        excluded_logical_class: np.ndarray,
        num_total_logical_classes: int,
        coverage_fraction: float | None = None,
        logical_error_distribution: np.ndarray | None = None,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Sample random logical classes excluding a given logical class.

        Parameters
        ----------
        excluded_logical_class : 1D numpy array of bool
            Logical class to exclude from sampling (typically the initial best class).
        num_total_logical_classes : int
            Total number of logical classes to explore including `excluded_logical_class`.
            Randomly samples `num_total_logical_classes - 1` additional classes.
        coverage_fraction : float, optional
            Fraction of cumulative probability mass to include when sampling.
            When specified (and < 1.0), only logical errors whose cumulative
            probability (sorted by likelihood) is <= coverage_fraction are
            eligible for sampling. Must be in (0, 1]. If 1.0 or None, samples
            uniformly from all classes. Requires `logical_error_distribution`
            when < 1.0. Defaults to None.
        logical_error_distribution : 1D numpy array of float, optional
            Distribution over logical errors with shape (2^k,). Index i
            corresponds to logical error with bit pattern i = sum(b_j * 2^j).
            Index 0 represents no error (identity). Required when
            `coverage_fraction` is specified and < 1.0. Defaults to None.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        sampled_logical_classes : list of 1D numpy array of bool
            Randomly sampled logical classes, excluding `excluded_logical_class`.
        """
        if num_total_logical_classes < 1:
            raise ValueError("explore_random_logical_classes must be >= 1")

        num_observables = len(excluded_logical_class)
        if num_observables == 0:
            return []

        total_num_logical_classes = 1 << num_observables
        max_additional_logical_classes = total_num_logical_classes - 1
        num_additional_logical_classes = min(
            num_total_logical_classes - 1, max_additional_logical_classes
        )

        if num_additional_logical_classes == 0:
            return []

        if (
            num_additional_logical_classes == max_additional_logical_classes
            and max_additional_logical_classes > 1_000_000
        ):
            raise ValueError(
                "Requested to explore all logical classes via explore_random_logical_classes, "
                "but the number of logical classes is too large."
            )

        # Check if coverage-restricted sampling should be used
        use_coverage_restriction = (
            coverage_fraction is not None and coverage_fraction < 1.0
        )

        if use_coverage_restriction:
            # Validate coverage_fraction
            if coverage_fraction <= 0:
                raise ValueError(
                    f"coverage_fraction must be in (0, 1], got {coverage_fraction}"
                )
            if logical_error_distribution is None:
                raise ValueError(
                    "logical_error_distribution must be provided when "
                    "coverage_fraction < 1.0 for 'random' method."
                )
            # Validate logical_error_distribution length
            expected_dist_len = 1 << num_observables
            if len(logical_error_distribution) != expected_dist_len:
                raise ValueError(
                    f"logical_error_distribution has length {len(logical_error_distribution)}, "
                    f"but expected {expected_dist_len} for {num_observables} observables."
                )

            # Use coverage-restricted sampling via error distribution
            return self._sample_coverage_restricted_logical_classes(
                best_logical_class=excluded_logical_class,
                logical_error_distribution=logical_error_distribution,
                num_classes_to_explore=num_total_logical_classes,
                coverage_fraction=coverage_fraction,
                verbose=verbose,
            )

        # Original uniform random sampling path
        # Integer representation where bit i corresponds to excluded_logical_class[i]
        if num_observables <= 64:
            # Vectorized path (fast)
            powers_of_two = np.uint64(1) << np.arange(num_observables, dtype=np.uint64)
            excluded_logical_class_int = int(
                excluded_logical_class.astype(np.uint64) @ powers_of_two
            )
        else:
            # Fallback for >64 observables (uses Python arbitrary-precision int)
            excluded_logical_class_int = 0
            for bit_index, bit in enumerate(excluded_logical_class.tolist()):
                if bit:
                    excluded_logical_class_int |= 1 << bit_index

        sampled_class_ints: List[int]
        if num_observables <= 62:
            base_samples = random.sample(
                range(total_num_logical_classes - 1), num_additional_logical_classes
            )
            sampled_class_ints = [
                x + 1 if x >= excluded_logical_class_int else x for x in base_samples
            ]
        else:
            selected: set[int] = set()
            while len(selected) < num_additional_logical_classes:
                candidate = random.getrandbits(num_observables)
                if candidate == excluded_logical_class_int:
                    continue
                selected.add(candidate)
            sampled_class_ints = list(selected)

        if num_observables <= 64:
            # Vectorized path for typical cases (fast)
            sampled_class_ints_arr = np.array(sampled_class_ints, dtype=np.uint64)[
                :, np.newaxis
            ]
            bit_positions = np.arange(num_observables, dtype=np.uint64)
            sampled_logical_classes_arr = (
                (sampled_class_ints_arr >> bit_positions) & 1
            ).astype(bool)
            sampled_logical_classes = [
                row.copy() for row in sampled_logical_classes_arr
            ]
        else:
            # Fallback for >64 observables (uses Python arbitrary-precision int)
            sampled_logical_classes = [
                np.array(
                    [
                        (logical_class_int >> bit_index) & 1
                        for bit_index in range(num_observables)
                    ],
                    dtype=bool,
                )
                for logical_class_int in sampled_class_ints
            ]

        if verbose:
            print(
                f"  Randomly exploring {len(sampled_logical_classes)} additional logical classes "
                f"(requested_total={num_total_logical_classes})"
            )

        return sampled_logical_classes

    def _sample_coverage_restricted_logical_classes(
        self,
        best_logical_class: np.ndarray,
        logical_error_distribution: np.ndarray,
        num_classes_to_explore: int,
        coverage_fraction: float,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Sample logical classes uniformly from errors within a coverage fraction.

        Restricts sampling to the most likely logical errors whose cumulative
        probability is <= coverage_fraction, then samples uniformly from that set.

        Parameters
        ----------
        best_logical_class : 1D numpy array of bool
            The best logical class from initial decoding.
        logical_error_distribution : 1D numpy array of float
            Distribution over logical errors with shape (2^k,).
            Index i corresponds to logical error with bit pattern i = sum(b_j * 2^j).
            Index 0 represents no error (identity).
        num_classes_to_explore : int
            Total number of logical classes to explore including the initial best class.
        coverage_fraction : float
            Fraction of cumulative probability mass to include.
            Only errors with cumulative probability <= coverage_fraction are eligible.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        logical_classes_to_explore : list of 1D numpy array of bool
            List of logical classes to explore (excluding the initial best class).
        """
        num_observables = len(best_logical_class)
        if num_observables == 0:
            return []

        total_num_logical_classes = 1 << num_observables
        num_additional_classes = min(
            num_classes_to_explore - 1, total_num_logical_classes - 1
        )

        if num_additional_classes <= 0:
            return []

        # Check cache for precomputed eligible error indices
        cache_key = (id(logical_error_distribution), coverage_fraction, num_observables)
        if cache_key in self._coverage_eligible_cache:
            eligible_error_indices = self._coverage_eligible_cache[cache_key]
        else:
            # Compute eligible error indices
            # Create array of valid error indices (exclude identity at index 0)
            valid_error_indices = np.arange(1, total_num_logical_classes)

            # Get weights for valid errors
            weights = logical_error_distribution[valid_error_indices].astype(float)
            weight_sum = weights.sum()

            if weight_sum <= 0:
                raise ValueError(
                    "All non-identity logical error weights sum to zero. "
                    "Coverage-restricted sampling requires a valid distribution "
                    "with positive weights for at least some non-identity errors. "
                    "Use coverage_fraction=None or 1.0 for uniform sampling without distribution."
                )

            # Sort by probability (descending)
            sorted_order = np.argsort(weights)[::-1]
            sorted_error_indices = valid_error_indices[sorted_order]
            sorted_weights = weights[sorted_order]

            # Normalize and compute cumulative probabilities
            normalized_probs = sorted_weights / weight_sum
            cumulative_probs = np.cumsum(normalized_probs)

            # Select errors with cumulative probability <= coverage_fraction
            eligible_mask = cumulative_probs <= coverage_fraction
            # Always include at least one error (the most likely one)
            if not np.any(eligible_mask):
                eligible_mask[0] = True
            eligible_error_indices = sorted_error_indices[eligible_mask]

            # Store in cache
            self._coverage_eligible_cache[cache_key] = eligible_error_indices

        # Validate that num_classes_to_explore covers all eligible errors
        # (num_additional_classes = num_classes_to_explore - 1, since initial class is excluded)
        num_eligible = len(eligible_error_indices)
        if num_additional_classes < num_eligible:
            raise ValueError(
                f"num_classes_to_explore ({num_classes_to_explore}) is smaller than the "
                f"number of eligible errors ({num_eligible} + 1 initial = {num_eligible + 1}) "
                f"at coverage_fraction={coverage_fraction}. Increase num_classes_to_explore "
                f"to at least {num_eligible + 1} to cover all eligible errors."
            )

        if verbose:
            print(
                f"  Coverage-restricted sampling: {num_eligible} "
                f"eligible errors (coverage_fraction={coverage_fraction})"
            )

        # Sample uniformly from eligible errors (all are taken since we validated above)
        num_to_sample = min(num_additional_classes, num_eligible)
        if num_to_sample == len(eligible_error_indices):
            # Take all eligible errors
            sampled_error_indices = eligible_error_indices
        else:
            # Uniform random sampling without replacement
            sampled_indices = random.sample(
                range(len(eligible_error_indices)), num_to_sample
            )
            sampled_error_indices = eligible_error_indices[sampled_indices]

        if verbose:
            print(
                f"  Sampling {num_to_sample} logical errors uniformly from eligible pool "
                f"(requested_total={num_classes_to_explore})"
            )

        # Convert error indices to logical classes by XOR with best_logical_class
        logical_classes_to_explore = []
        for error_idx in sampled_error_indices:
            error_pattern = self._index_to_logical_class(
                int(error_idx), num_observables
            )
            resulting_class = best_logical_class ^ error_pattern
            logical_classes_to_explore.append(resulting_class)

        return logical_classes_to_explore

    def _index_to_logical_class(
        self,
        index: int,
        num_observables: int,
    ) -> np.ndarray:
        """
        Convert an integer index to a logical class bit pattern.

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

    def _get_most_likely_logical_classes(
        self,
        best_logical_class: np.ndarray,
        logical_error_distribution: np.ndarray,
        num_classes_to_explore: int,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Get logical classes to explore based on most likely logical errors.

        Selects the most probable logical errors from the distribution
        (excluding identity), applies XOR with best_logical_class to obtain
        the corresponding logical classes.

        Parameters
        ----------
        best_logical_class : 1D numpy array of bool
            The best logical class from initial decoding.
        logical_error_distribution : 1D numpy array of float
            Distribution over logical errors with shape (2^k,).
            Index i corresponds to logical error with bit pattern i = sum(b_j * 2^j).
            Index 0 represents no error (identity). Higher values indicate more
            probable logical errors.
        num_classes_to_explore : int
            Total number of logical classes to explore including the initial best class.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        logical_classes_to_explore : list of 1D numpy array of bool
            List of logical classes to explore (excluding the initial best class).
        """
        num_observables = len(best_logical_class)
        if num_observables == 0:
            return []

        total_num_logical_classes = 1 << num_observables
        num_additional_classes = min(
            num_classes_to_explore - 1, total_num_logical_classes - 1
        )

        if num_additional_classes <= 0:
            return []

        # Get indices of logical errors sorted by probability (descending)
        # Exclude index 0 (identity/no error) since applying it gives the same class
        error_indices_sorted = np.argsort(logical_error_distribution)[::-1]

        # Filter out index 0 (identity) and take top num_additional_classes
        non_identity_sorted = error_indices_sorted[error_indices_sorted != 0]
        selected_error_indices = non_identity_sorted[:num_additional_classes]

        if verbose:
            print(
                f"  Selecting {len(selected_error_indices)} most likely logical errors "
                f"(requested_total={num_classes_to_explore})"
            )
            if len(selected_error_indices) > 0:
                selected_probs = logical_error_distribution[selected_error_indices]
                print(
                    f"    Top error indices: {selected_error_indices[:5].tolist()}..."
                )
                print(f"    Corresponding values: {selected_probs[:5].tolist()}...")

        # Convert error indices to logical classes by XOR with best_logical_class
        logical_classes_to_explore = []

        for error_idx in selected_error_indices:
            # Convert error index to bit pattern
            error_pattern = self._index_to_logical_class(
                int(error_idx), num_observables
            )
            # Apply error to best class (XOR)
            resulting_class = best_logical_class ^ error_pattern
            logical_classes_to_explore.append(resulting_class)

        return logical_classes_to_explore

    def _sample_weighted_random_logical_classes(
        self,
        best_logical_class: np.ndarray,
        logical_error_distribution: np.ndarray,
        num_classes_to_explore: int,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Sample logical classes to explore using weighted random sampling.

        Samples logical errors from the distribution (excluding identity),
        applies XOR with best_logical_class to obtain logical classes.
        Uses sampling without replacement with probabilities proportional
        to the distribution values.

        Parameters
        ----------
        best_logical_class : 1D numpy array of bool
            The best logical class from initial decoding.
        logical_error_distribution : 1D numpy array of float
            Distribution over logical errors with shape (2^k,).
            Index i corresponds to logical error with bit pattern i = sum(b_j * 2^j).
            Index 0 represents no error (identity). Can be raw counts or probabilities;
            values will be normalized internally.
        num_classes_to_explore : int
            Total number of logical classes to explore including the initial best class.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        logical_classes_to_explore : list of 1D numpy array of bool
            List of logical classes to explore (excluding the initial best class).
        """
        num_observables = len(best_logical_class)
        if num_observables == 0:
            return []

        total_num_logical_classes = 1 << num_observables
        num_additional_classes = min(
            num_classes_to_explore - 1, total_num_logical_classes - 1
        )

        if num_additional_classes <= 0:
            return []

        # Create array of valid error indices (exclude identity at index 0)
        valid_error_indices = np.arange(1, total_num_logical_classes)

        # Get weights for valid errors and normalize to probabilities
        weights = logical_error_distribution[valid_error_indices].astype(float)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            # Fall back to uniform if all weights are zero
            probabilities = np.ones(len(weights)) / len(weights)
            if verbose:
                print(
                    "  Warning: All distribution weights are zero, "
                    "falling back to uniform sampling"
                )
        else:
            probabilities = weights / weight_sum

        # Sample without replacement using weighted probabilities
        num_to_sample = min(num_additional_classes, len(valid_error_indices))
        sampled_error_indices = np.random.choice(
            valid_error_indices,
            size=num_to_sample,
            replace=False,
            p=probabilities,
        )

        if verbose:
            print(
                f"  Sampling {num_to_sample} logical errors using weighted-random "
                f"(requested_total={num_classes_to_explore})"
            )
            if num_to_sample > 0:
                sampled_probs = logical_error_distribution[sampled_error_indices]
                print(
                    f"    Sampled error indices: {sampled_error_indices[:5].tolist()}..."
                )
                print(f"    Corresponding values: {sampled_probs[:5].tolist()}...")

        # Convert error indices to logical classes by XOR with best_logical_class
        logical_classes_to_explore = []
        for error_idx in sampled_error_indices:
            # Convert error index to bit pattern
            error_pattern = self._index_to_logical_class(
                int(error_idx), num_observables
            )
            # Apply error to best class (XOR)
            resulting_class = best_logical_class ^ error_pattern
            logical_classes_to_explore.append(resulting_class)

        return logical_classes_to_explore

    def _get_next_mlf_adaptive_class(
        self,
        current_best_class: np.ndarray,
        sorted_error_indices: np.ndarray,
        explored_classes_set: set,
        start_index: int = 0,
    ) -> tuple[np.ndarray | None, int]:
        """
        Get the next unexplored logical class for most-likely-first-adaptive.

        Iterates through errors sorted by probability (descending), starting from
        start_index. Returns the first class (current_best XOR error) not already
        explored, along with the next start index for subsequent calls.

        Parameters
        ----------
        current_best_class : 1D numpy array of bool
            The current best logical class (offset for XOR).
        sorted_error_indices : 1D numpy array of int
            Error indices sorted by probability in descending order.
        explored_classes_set : set of tuple
            Set of already explored logical classes as tuples.
        start_index : int, optional
            Index in sorted_error_indices to start searching from. Defaults to 0.
            Should be reset to 0 when current_best_class changes.

        Returns
        -------
        next_class : 1D numpy array of bool or None
            The next unexplored logical class, or None if all classes have been explored.
        next_start_index : int
            The index to use as start_index for the next call (if best class unchanged).
        """
        num_observables = len(current_best_class)
        n_indices = len(sorted_error_indices)

        for i in range(start_index, n_indices):
            error_idx = sorted_error_indices[i]
            if error_idx == 0:
                continue  # Skip identity
            error_pattern = self._index_to_logical_class(
                int(error_idx), num_observables
            )
            candidate_class = current_best_class ^ error_pattern
            if tuple(candidate_class) not in explored_classes_set:
                return candidate_class, i + 1

        return None, n_indices

    def _sample_next_wr_adaptive_class(
        self,
        current_best_class: np.ndarray,
        valid_error_indices: np.ndarray,
        probabilities: np.ndarray,
        explored_classes_set: set,
        max_retries: int = 1000,
    ) -> np.ndarray | None:
        """
        Sample the next unexplored logical class for weighted-random-adaptive.

        Uses rejection sampling: samples error from distribution, rejects if
        resulting class already explored.

        Parameters
        ----------
        current_best_class : 1D numpy array of bool
            The current best logical class (offset for XOR).
        valid_error_indices : 1D numpy array of int
            Valid error indices (excluding identity).
        probabilities : 1D numpy array of float
            Normalized probabilities for each valid error index.
        explored_classes_set : set of tuple
            Set of already explored logical classes as tuples.
        max_retries : int, optional
            Maximum number of sampling attempts before giving up. Defaults to 1000.

        Returns
        -------
        next_class : 1D numpy array of bool or None
            The next unexplored logical class, or None if max_retries exceeded.
        """
        num_observables = len(current_best_class)

        for _ in range(max_retries):
            error_idx = np.random.choice(valid_error_indices, p=probabilities)
            error_pattern = self._index_to_logical_class(
                int(error_idx), num_observables
            )
            candidate_class = current_best_class ^ error_pattern
            if tuple(candidate_class) not in explored_classes_set:
                return candidate_class

        return None  # Failed to find unexplored class

    def _perform_fixed_logical_class_decoding(
        self,
        detector_outcomes: np.ndarray,
        fixed_logical_class: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform fixed-logical-class decoding for a given logical class.

        Uses cached decoder for efficiency when exploring multiple logical classes.

        Parameters
        ----------
        detector_outcomes : 1D numpy array of bool
            Detector measurement outcomes.
        fixed_logical_class : 1D numpy array of bool
            Fixed logical class to decode with.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        pred_llr : float
            Prediction LLR for the fixed logical class decoding.
        pred : 1D numpy array of bool
            Predicted error pattern for the fixed logical class decoding.
        """
        if verbose:
            print(
                f"    Performing fixed-logical-class decoding for class {fixed_logical_class}"
            )

        # Use cached decoder with explicit guard
        decoder_fixed = self._decoder_for_fixed_class
        if decoder_fixed is None:
            raise ValueError(
                "Cannot perform fixed-logical-class decoding without obs_matrix. "
                "Ensure obs_matrix is provided during decoder initialization."
            )

        # Construct detector_outcomes_obs_appended
        detector_outcomes_obs_appended = np.concatenate(
            [detector_outcomes, fixed_logical_class]
        )

        # Perform decoding without cluster stats computation for efficiency
        pred_fixed, _, _, soft_outputs_fixed = decoder_fixed.decode(
            detector_outcomes_obs_appended,
            include_cluster_stats=False,
            compute_logical_gap_proxy=False,  # Avoid recursion
            verbose=False,  # Don't cascade verbose to avoid excessive output
        )

        if verbose:
            print(
                f"    Fixed-class decoding completed, pred_llr={soft_outputs_fixed['pred_llr']:.4f}"
            )

        return soft_outputs_fixed["pred_llr"], pred_fixed

    def _compute_logical_gap_proxy(
        self,
        detector_outcomes: np.ndarray,
        pred: np.ndarray,
        original_pred_llr: float,
        logical_gap_proxy_method: str | None,
        num_classes_to_explore: int | None = None,
        compute_all_intermediate_gap_proxies: bool = False,
        logical_error_distribution: np.ndarray | None = None,
        coverage_fraction: float | None = None,
        num_procs_for_gap: int = 1,
        return_explored_classes: bool = False,
        parallel_verbose: int = 0,
        verbose: bool = False,
    ) -> Tuple[
        float,
        np.ndarray,
        float,
        Dict[int, float],
        Dict[Tuple[bool, ...], Tuple[float, np.ndarray]] | None,
    ]:
        """
        Compute logical gap proxy by exploring different logical classes iteratively.

        Parameters
        ----------
        detector_outcomes : 1D numpy array of bool
            Detector measurement outcomes.
        pred : 1D numpy array of bool
            Original predicted error pattern.
        original_pred_llr : float
            Original prediction LLR.
        logical_gap_proxy_method : str or None
            Method for exploring logical classes:
            - None: Explore all possible logical classes (exact gap proxy).
            - 'nearby': Only explore nearby logical classes (flip one bit at a time).
            - 'random': Randomly sample logical classes uniformly.
            - 'most-likely-first': Deterministically select top classes by distribution.
            - 'weighted-random': Sample classes with probabilities from distribution.
            - 'most-likely-first-adaptive': Like most-likely-first but updates base
              class when better class found during exploration.
            - 'weighted-random-adaptive': Like weighted-random but updates base
              class when better class found during exploration.
        num_classes_to_explore : int, optional
            Total number of logical classes to explore including the initial best class.
            Required when `logical_gap_proxy_method` is 'random', 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive'.
        compute_all_intermediate_gap_proxies : bool, optional
            If True and `logical_gap_proxy_method` is 'random', 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive',
            compute and store gap proxies for all intermediate numbers of explored logical
            classes. Defaults to False.
        logical_error_distribution : 1D numpy array of float, optional
            Distribution over logical errors with shape (2^k,) where k is the number
            of observables. Index i corresponds to logical error with bit pattern
            i = sum(b_j * 2^j) for j=0..k-1. Higher values indicate more probable
            errors. Required when `logical_gap_proxy_method` is 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive'.
            Also required when `logical_gap_proxy_method` is 'random' and
            `coverage_fraction` is specified and < 1.0. Values are auto-normalized internally.
        coverage_fraction : float, optional
            Fraction of cumulative probability mass to include when sampling
            logical classes for the 'random' gap proxy method. When specified
            (and < 1.0), only logical errors whose cumulative probability (sorted
            by likelihood) is <= coverage_fraction are eligible for uniform sampling.
            Must be in (0, 1]. If 1.0 or None, samples uniformly from all classes
            (default behavior). Requires `logical_error_distribution` when < 1.0.
            Only used when `logical_gap_proxy_method` is 'random'.
            Defaults to None.
        num_procs_for_gap : int, optional
            Number of parallel processes for decoding logical classes. Only effective
            for methods where classes are determined upfront: None (exhaustive),
            'random', 'most-likely-first', 'weighted-random'. Set to -1 to use
            all available CPUs. Raises ValueError for 'nearby' and adaptive methods
            when > 1. Defaults to 1 (sequential).
        return_explored_classes : bool, optional
            If True, return the explored_classes dictionary containing all explored
            logical classes and their results. Useful for detailed analysis of the
            decoding results across all logical classes. Defaults to False.
        parallel_verbose : int, optional
            Verbosity level for joblib parallel execution. 0 means silent, higher
            values show progress information. Passed directly to joblib.Parallel.
            Defaults to 0.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        gap_proxy : float
            Logical gap proxy (difference between minimum and second minimum pred_llr).
        best_pred : 1D numpy array of bool
            Best predicted error pattern (corresponding to minimum pred_llr).
        best_pred_llr : float
            Best prediction LLR (minimum among all explored).
        gap_proxies_by_num_classes : dict of int to float
            Dictionary mapping the number of explored logical classes to the
            corresponding gap proxy. Only populated when `compute_all_intermediate_gap_proxies`
            is True and `logical_gap_proxy_method` is 'random', 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive'.
        explored_classes : dict or None
            Dictionary mapping logical class tuples to (pred_llr, pred_pattern) tuples.
            Only returned (non-None) when `return_explored_classes` is True.
            Keys are tuples of bool representing the logical class.
            Values are tuples of (float, numpy array) for (pred_llr, pred_pattern).
        """
        # Validate num_procs_for_gap for methods that require sequential execution
        if num_procs_for_gap != 1 and logical_gap_proxy_method in (
            "nearby",
            "most-likely-first-adaptive",
            "weighted-random-adaptive",
        ):
            raise ValueError(
                f"num_procs_for_gap={num_procs_for_gap} is not supported "
                f"for logical_gap_proxy_method='{logical_gap_proxy_method}' which "
                f"requires sequential execution due to adaptive exploration."
            )

        if verbose:
            print("  Computing logical gap proxy...")

        if self.obs_matrix is None:
            if verbose:
                print("  No observables, returning original results")
            return 0.0, pred, original_pred_llr, {}, None

        # Validate logical_gap_proxy_method
        if logical_gap_proxy_method is not None and logical_gap_proxy_method not in (
            "nearby",
            "random",
            "most-likely-first",
            "weighted-random",
            "most-likely-first-adaptive",
            "weighted-random-adaptive",
        ):
            raise ValueError(
                f"Invalid logical_gap_proxy_method: {logical_gap_proxy_method}. "
                "Must be None, 'nearby', 'random', 'most-likely-first', 'weighted-random', "
                "'most-likely-first-adaptive', or 'weighted-random-adaptive'."
            )

        # Validate num_classes_to_explore for 'random' method
        if logical_gap_proxy_method == "random":
            if num_classes_to_explore is None:
                raise ValueError(
                    "num_classes_to_explore must be provided when "
                    "logical_gap_proxy_method is 'random'."
                )
            if num_classes_to_explore < 1:
                raise ValueError("num_classes_to_explore must be >= 1")

        # Validate parameters for methods that require distribution
        if logical_gap_proxy_method in (
            "most-likely-first",
            "weighted-random",
            "most-likely-first-adaptive",
            "weighted-random-adaptive",
        ):
            if logical_error_distribution is None:
                raise ValueError(
                    "logical_error_distribution must be provided when "
                    f"logical_gap_proxy_method is '{logical_gap_proxy_method}'."
                )
            if num_classes_to_explore is None:
                raise ValueError(
                    "num_classes_to_explore must be provided when "
                    f"logical_gap_proxy_method is '{logical_gap_proxy_method}'."
                )
            if num_classes_to_explore < 1:
                raise ValueError("num_classes_to_explore must be >= 1")
            num_observables = self.obs_matrix.shape[0]
            expected_dist_len = 1 << num_observables
            if len(logical_error_distribution) != expected_dist_len:
                raise ValueError(
                    f"logical_error_distribution has length {len(logical_error_distribution)}, "
                    f"but expected {expected_dist_len} for {num_observables} observables."
                )

        # Calculate original logical class
        original_logical_class = (
            (pred.astype("uint8") @ self.obs_matrix.T) % 2
        ).astype(bool)

        if verbose:
            print(f"  Original logical class: {original_logical_class}")

        # Store all explored logical classes and their results
        explored_classes = {}  # logical_class tuple -> (pred_llr, pred_pattern)
        explored_classes[tuple(original_logical_class)] = (original_pred_llr, pred)

        # Initialize intermediate gap proxies dict (used by random/most-likely-first/weighted-random)
        gap_proxies_by_num_classes: Dict[int, float] = {}

        if logical_gap_proxy_method == "random":
            random_logical_classes = self._sample_random_logical_classes(
                excluded_logical_class=original_logical_class,
                num_total_logical_classes=num_classes_to_explore,
                coverage_fraction=coverage_fraction,
                logical_error_distribution=logical_error_distribution,
                verbose=verbose,
            )

            if num_procs_for_gap == 1:
                # Sequential execution
                if compute_all_intermediate_gap_proxies:
                    running_best_llr = float(original_pred_llr)
                    running_second_best_llr = float("inf")
                    explored_count = 1

                for logical_class in random_logical_classes:
                    pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                        detector_outcomes, logical_class, verbose=verbose
                    )
                    explored_classes[tuple(logical_class)] = (pred_llr, pred_pattern)

                    if compute_all_intermediate_gap_proxies:
                        explored_count += 1
                        pred_llr_float = float(pred_llr)
                        if pred_llr_float <= running_best_llr:
                            running_second_best_llr = running_best_llr
                            running_best_llr = pred_llr_float
                        elif pred_llr_float < running_second_best_llr:
                            running_second_best_llr = pred_llr_float

                        if explored_count >= 2:
                            effective_second = (
                                running_second_best_llr
                                if running_second_best_llr != float("inf")
                                else running_best_llr
                            )
                            gap_proxies_by_num_classes[explored_count] = float(
                                effective_second - running_best_llr
                            )
            else:
                # Parallel execution
                if verbose:
                    print(
                        f"  Parallel decoding with num_procs_for_gap="
                        f"{num_procs_for_gap}"
                    )

                H_obs_appended = self._H_obs_appended
                priors = self.priors
                bplsd_kwargs = self._bplsd_kwargs.copy()

                results = Parallel(
                    n_jobs=num_procs_for_gap,
                    prefer="processes",
                    verbose=parallel_verbose,
                )(
                    delayed(_decode_single_logical_class)(
                        H_obs_appended, priors, bplsd_kwargs, detector_outcomes, lc
                    )
                    for lc in random_logical_classes
                )

                for lc_tuple, pred_llr, pred_pattern in results:
                    explored_classes[lc_tuple] = (pred_llr, pred_pattern)

                if compute_all_intermediate_gap_proxies:
                    gap_proxies_by_num_classes = (
                        _compute_intermediate_gap_proxies_posthoc(
                            original_pred_llr, results
                        )
                    )

        elif logical_gap_proxy_method == "most-likely-first":
            # Get logical classes to explore based on most likely logical errors
            most_likely_logical_classes = self._get_most_likely_logical_classes(
                best_logical_class=original_logical_class,
                logical_error_distribution=logical_error_distribution,
                num_classes_to_explore=num_classes_to_explore,
                verbose=verbose,
            )

            if num_procs_for_gap == 1:
                # Sequential execution
                if compute_all_intermediate_gap_proxies:
                    running_best_llr = float(original_pred_llr)
                    running_second_best_llr = float("inf")
                    explored_count = 1

                for logical_class in most_likely_logical_classes:
                    pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                        detector_outcomes, logical_class, verbose=verbose
                    )
                    explored_classes[tuple(logical_class)] = (pred_llr, pred_pattern)

                    if compute_all_intermediate_gap_proxies:
                        explored_count += 1
                        pred_llr_float = float(pred_llr)
                        if pred_llr_float <= running_best_llr:
                            running_second_best_llr = running_best_llr
                            running_best_llr = pred_llr_float
                        elif pred_llr_float < running_second_best_llr:
                            running_second_best_llr = pred_llr_float

                        if explored_count >= 2:
                            effective_second = (
                                running_second_best_llr
                                if running_second_best_llr != float("inf")
                                else running_best_llr
                            )
                            gap_proxies_by_num_classes[explored_count] = float(
                                effective_second - running_best_llr
                            )
            else:
                # Parallel execution
                if verbose:
                    print(
                        f"  Parallel decoding with num_procs_for_gap="
                        f"{num_procs_for_gap}"
                    )

                H_obs_appended = self._H_obs_appended
                priors = self.priors
                bplsd_kwargs = self._bplsd_kwargs.copy()

                results = Parallel(
                    n_jobs=num_procs_for_gap,
                    prefer="processes",
                    verbose=parallel_verbose,
                )(
                    delayed(_decode_single_logical_class)(
                        H_obs_appended, priors, bplsd_kwargs, detector_outcomes, lc
                    )
                    for lc in most_likely_logical_classes
                )

                for lc_tuple, pred_llr, pred_pattern in results:
                    explored_classes[lc_tuple] = (pred_llr, pred_pattern)

                if compute_all_intermediate_gap_proxies:
                    gap_proxies_by_num_classes = (
                        _compute_intermediate_gap_proxies_posthoc(
                            original_pred_llr, results
                        )
                    )

        elif logical_gap_proxy_method == "weighted-random":
            # Get logical classes to explore using weighted random sampling
            weighted_random_logical_classes = (
                self._sample_weighted_random_logical_classes(
                    best_logical_class=original_logical_class,
                    logical_error_distribution=logical_error_distribution,
                    num_classes_to_explore=num_classes_to_explore,
                    verbose=verbose,
                )
            )

            if num_procs_for_gap == 1:
                # Sequential execution
                if compute_all_intermediate_gap_proxies:
                    running_best_llr = float(original_pred_llr)
                    running_second_best_llr = float("inf")
                    explored_count = 1

                for logical_class in weighted_random_logical_classes:
                    pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                        detector_outcomes, logical_class, verbose=verbose
                    )
                    explored_classes[tuple(logical_class)] = (pred_llr, pred_pattern)

                    if compute_all_intermediate_gap_proxies:
                        explored_count += 1
                        pred_llr_float = float(pred_llr)
                        if pred_llr_float <= running_best_llr:
                            running_second_best_llr = running_best_llr
                            running_best_llr = pred_llr_float
                        elif pred_llr_float < running_second_best_llr:
                            running_second_best_llr = pred_llr_float

                        if explored_count >= 2:
                            effective_second = (
                                running_second_best_llr
                                if running_second_best_llr != float("inf")
                                else running_best_llr
                            )
                            gap_proxies_by_num_classes[explored_count] = float(
                                effective_second - running_best_llr
                            )
            else:
                # Parallel execution
                if verbose:
                    print(
                        f"  Parallel decoding with num_procs_for_gap="
                        f"{num_procs_for_gap}"
                    )

                H_obs_appended = self._H_obs_appended
                priors = self.priors
                bplsd_kwargs = self._bplsd_kwargs.copy()

                results = Parallel(
                    n_jobs=num_procs_for_gap,
                    prefer="processes",
                    verbose=parallel_verbose,
                )(
                    delayed(_decode_single_logical_class)(
                        H_obs_appended, priors, bplsd_kwargs, detector_outcomes, lc
                    )
                    for lc in weighted_random_logical_classes
                )

                for lc_tuple, pred_llr, pred_pattern in results:
                    explored_classes[lc_tuple] = (pred_llr, pred_pattern)

                if compute_all_intermediate_gap_proxies:
                    gap_proxies_by_num_classes = (
                        _compute_intermediate_gap_proxies_posthoc(
                            original_pred_llr, results
                        )
                    )

        elif logical_gap_proxy_method == "most-likely-first-adaptive":
            # Adaptive exploration: update base class when better class found
            # Pre-sort error indices by distribution (descending probability)
            sorted_error_indices = np.argsort(logical_error_distribution)[::-1]

            # Track explored classes as set for O(1) lookup
            explored_classes_set = {tuple(original_logical_class)}
            current_best_class = original_logical_class.copy()
            current_best_llr = float(original_pred_llr)

            # Cursor for efficient scanning (reset to 0 when best class changes)
            search_cursor = 0

            # Initialize intermediate tracking
            if compute_all_intermediate_gap_proxies:
                running_best_llr = float(original_pred_llr)
                running_second_best_llr = float("inf")
                explored_count = 1

            if verbose:
                print(
                    f"  Exploring up to {num_classes_to_explore} logical classes "
                    f"using most-likely-first-adaptive"
                )

            # Explore until we have num_classes_to_explore classes
            while len(explored_classes_set) < num_classes_to_explore:
                next_class, search_cursor = self._get_next_mlf_adaptive_class(
                    current_best_class,
                    sorted_error_indices,
                    explored_classes_set,
                    start_index=search_cursor,
                )
                if next_class is None:
                    if verbose:
                        print(
                            f"    No more unexplored classes available "
                            f"(explored {len(explored_classes_set)} classes)"
                        )
                    break  # No more unexplored classes

                # Decode and store
                pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                    detector_outcomes, next_class, verbose=verbose
                )
                explored_classes[tuple(next_class)] = (pred_llr, pred_pattern)
                explored_classes_set.add(tuple(next_class))

                # Update current best if this class is better
                pred_llr_float = float(pred_llr)
                if pred_llr_float < current_best_llr:
                    if verbose:
                        print(
                            f"    New best class found: {next_class} "
                            f"(llr={pred_llr_float:.4f} < {current_best_llr:.4f})"
                        )
                    current_best_class = next_class.copy()
                    current_best_llr = pred_llr_float
                    # Reset cursor when best class changes (new base = new candidates)
                    search_cursor = 0

                # Track intermediate gap proxies if requested
                if compute_all_intermediate_gap_proxies:
                    explored_count += 1
                    if pred_llr_float <= running_best_llr:
                        running_second_best_llr = running_best_llr
                        running_best_llr = pred_llr_float
                    elif pred_llr_float < running_second_best_llr:
                        running_second_best_llr = pred_llr_float

                    if explored_count >= 2:
                        effective_second = (
                            running_second_best_llr
                            if running_second_best_llr != float("inf")
                            else running_best_llr
                        )
                        gap_proxies_by_num_classes[explored_count] = float(
                            effective_second - running_best_llr
                        )

        elif logical_gap_proxy_method == "weighted-random-adaptive":
            # Adaptive exploration with weighted random sampling
            num_observables = len(original_logical_class)
            total_num_logical_classes = 1 << num_observables

            # Prepare distribution (exclude identity)
            valid_error_indices = np.arange(1, total_num_logical_classes)
            weights = logical_error_distribution[valid_error_indices].astype(float)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                probabilities = np.ones(len(weights)) / len(weights)
                if verbose:
                    print(
                        "  Warning: All distribution weights are zero, "
                        "falling back to uniform sampling"
                    )
            else:
                probabilities = weights / weight_sum

            # Track explored classes as set for O(1) lookup
            explored_classes_set = {tuple(original_logical_class)}
            current_best_class = original_logical_class.copy()
            current_best_llr = float(original_pred_llr)

            # Initialize intermediate tracking
            if compute_all_intermediate_gap_proxies:
                running_best_llr = float(original_pred_llr)
                running_second_best_llr = float("inf")
                explored_count = 1

            if verbose:
                print(
                    f"  Exploring up to {num_classes_to_explore} logical classes "
                    f"using weighted-random-adaptive"
                )

            # Explore until we have num_classes_to_explore classes
            while len(explored_classes_set) < num_classes_to_explore:
                next_class = self._sample_next_wr_adaptive_class(
                    current_best_class,
                    valid_error_indices,
                    probabilities,
                    explored_classes_set,
                )
                if next_class is None:
                    if verbose:
                        print(
                            f"    Failed to sample unexplored class after max retries "
                            f"(explored {len(explored_classes_set)} classes)"
                        )
                    break  # Failed to find unexplored class

                # Decode and store
                pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                    detector_outcomes, next_class, verbose=verbose
                )
                explored_classes[tuple(next_class)] = (pred_llr, pred_pattern)
                explored_classes_set.add(tuple(next_class))

                # Update current best if this class is better
                pred_llr_float = float(pred_llr)
                if pred_llr_float < current_best_llr:
                    if verbose:
                        print(
                            f"    New best class found: {next_class} "
                            f"(llr={pred_llr_float:.4f} < {current_best_llr:.4f})"
                        )
                    current_best_class = next_class.copy()
                    current_best_llr = pred_llr_float

                # Track intermediate gap proxies if requested
                if compute_all_intermediate_gap_proxies:
                    explored_count += 1
                    if pred_llr_float <= running_best_llr:
                        running_second_best_llr = running_best_llr
                        running_best_llr = pred_llr_float
                    elif pred_llr_float < running_second_best_llr:
                        running_second_best_llr = pred_llr_float

                    if explored_count >= 2:
                        effective_second = (
                            running_second_best_llr
                            if running_second_best_llr != float("inf")
                            else running_best_llr
                        )
                        gap_proxies_by_num_classes[explored_count] = float(
                            effective_second - running_best_llr
                        )

        elif logical_gap_proxy_method is None:
            # Explore all classes (except the initial one)
            num_observables = len(original_logical_class)
            original_class_tuple = tuple(original_logical_class)

            if verbose:
                print(f"  Exploring all {1 << num_observables} logical classes")

            # Build list of all logical classes to explore (excluding original)
            all_logical_classes_to_explore = [
                np.array(lc_tuple, dtype=bool)
                for lc_tuple in product([False, True], repeat=num_observables)
                if lc_tuple != original_class_tuple
            ]

            if num_procs_for_gap == 1:
                # Sequential execution
                for logical_class in all_logical_classes_to_explore:
                    if verbose:
                        print(f"  Processing logical class {logical_class}")
                    pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                        detector_outcomes, logical_class, verbose=verbose
                    )
                    explored_classes[tuple(logical_class)] = (pred_llr, pred_pattern)
            else:
                # Parallel execution
                if verbose:
                    print(
                        f"  Parallel decoding with num_procs_for_gap="
                        f"{num_procs_for_gap}"
                    )

                H_obs_appended = self._H_obs_appended
                priors = self.priors
                bplsd_kwargs = self._bplsd_kwargs.copy()

                results = Parallel(
                    n_jobs=num_procs_for_gap,
                    prefer="processes",
                    verbose=parallel_verbose,
                )(
                    delayed(_decode_single_logical_class)(
                        H_obs_appended, priors, bplsd_kwargs, detector_outcomes, lc
                    )
                    for lc in all_logical_classes_to_explore
                )

                for lc_tuple, pred_llr, pred_pattern in results:
                    explored_classes[lc_tuple] = (pred_llr, pred_pattern)

        else:
            # 'nearby' method: Iterative exploration for nearby classes only
            to_explore = [original_logical_class]
            explored_set = {tuple(original_logical_class)}

            iteration = 0
            while to_explore:
                iteration += 1
                if verbose:
                    print(
                        f"  Iteration {iteration}: {len(to_explore)} classes to explore"
                    )

                current_class = to_explore.pop(0)

                # Get nearby logical classes (explore_only_nearby=True)
                nearby_classes = self._get_logical_classes_to_explore(
                    current_class,
                    explore_only_nearby_logical_classes=True,
                    verbose=False,
                )

                new_best_found = False
                current_best_llr = min(llr for llr, _ in explored_classes.values())

                for logical_class in nearby_classes:
                    logical_class_tuple = tuple(logical_class)
                    if logical_class_tuple not in explored_set:
                        if verbose:
                            print(f"    Processing nearby class {logical_class}")

                        pred_llr, pred_pattern = (
                            self._perform_fixed_logical_class_decoding(
                                detector_outcomes, logical_class, verbose=verbose
                            )
                        )
                        explored_classes[logical_class_tuple] = (pred_llr, pred_pattern)
                        explored_set.add(logical_class_tuple)

                        # If this is better than current best, add it to exploration queue
                        if pred_llr < current_best_llr:
                            to_explore.append(logical_class)
                            new_best_found = True
                            current_best_llr = pred_llr
                            if verbose:
                                print(f"    New best found: {pred_llr:.4f}")

                if verbose:
                    print(
                        f"  Iteration {iteration} completed, new best found: {new_best_found}"
                    )

        # Find best and second best
        all_pred_llrs = [llr for llr, _ in explored_classes.values()]
        all_pred_llrs.sort()

        best_pred_llr = all_pred_llrs[0]
        second_best_pred_llr = (
            all_pred_llrs[1] if len(all_pred_llrs) > 1 else best_pred_llr
        )

        # Find the logical class corresponding to best pred_llr
        best_logical_class_tuple = None
        for logical_class_tuple, (pred_llr, _) in explored_classes.items():
            if pred_llr == best_pred_llr:
                best_logical_class_tuple = logical_class_tuple
                break

        best_pred = explored_classes[best_logical_class_tuple][1]
        gap_proxy = second_best_pred_llr - best_pred_llr

        if verbose:
            print(f"  Total logical classes explored: {len(explored_classes)}")
            print(f"  Best pred_llr: {best_pred_llr:.4f}")
            print(f"  Second best pred_llr: {second_best_pred_llr:.4f}")
            print(f"  Gap proxy: {gap_proxy:.4f}")
            print(f"  Best logical class: {np.array(best_logical_class_tuple)}")

        return (
            gap_proxy,
            best_pred,
            best_pred_llr,
            gap_proxies_by_num_classes,
            explored_classes if return_explored_classes else None,
        )

    def _hash_matrix_and_priors(self, H_matrix: csc_matrix, priors: np.ndarray) -> str:
        """
        Generate a hash key for a matrix and priors combination.

        Parameters
        ----------
        H_matrix : scipy csc_matrix
            The parity check matrix.
        priors : 1D numpy array of float
            The error probabilities.

        Returns
        -------
        hash_key : str
            Hash string representing the matrix and priors configuration.
        """
        # Create a hash based on matrix structure and priors
        hasher = hashlib.md5()

        # Hash matrix shape
        hasher.update(str(H_matrix.shape).encode())

        # Hash matrix data (indices and indptr are sufficient for structure)
        hasher.update(H_matrix.indices.tobytes())
        hasher.update(H_matrix.indptr.tobytes())

        # Hash priors
        hasher.update(priors.tobytes())

        return hasher.hexdigest()

    def _get_or_create_window_structure(
        self, window_size: int, commit_size: int, window_position: int
    ) -> Dict[str, Any]:
        """
        Get or create cached window structure for a given window configuration.

        Parameters
        ----------
        window_size : int
            Number of rounds in each window.
        commit_size : int
            Number of rounds for each commitment.
        window_position : int
            The window index (w).

        Returns
        -------
        window_structure : dict
            Cached window structure containing:
            - window_detector_mask: boolean mask for detectors in this window
            - H_window_base: H matrix rows for this window (before fault filtering)
            - window_start: start time of window
            - window_end: end time of window
        """
        cache_key = (window_size, commit_size, window_position)

        if cache_key in self._window_structure_cache:
            return self._window_structure_cache[cache_key]

        # Compute window structure
        detector_times = self.detector_time_coords
        window_start = window_position * commit_size
        window_end = window_position * commit_size + window_size - 1

        # Extract detectors within window time range
        window_detector_mask = (detector_times >= window_start) & (
            detector_times <= window_end
        )

        # Extract corresponding rows from H matrix
        H_window_base = self.H[window_detector_mask, :]

        window_structure = {
            "window_detector_mask": window_detector_mask,
            "H_window_base": H_window_base,
            "window_start": window_start,
            "window_end": window_end,
        }

        # Cache the structure
        self._window_structure_cache[cache_key] = window_structure
        return window_structure

    def _get_or_create_window_decoder(
        self, H_window: csc_matrix, p_window: np.ndarray
    ) -> Self:
        """
        Get or create a cached decoder for the given H matrix and priors.

        Parameters
        ----------
        H_window : scipy csc_matrix
            The window parity check matrix.
        p_window : 1D numpy array of float
            The window error probabilities.

        Returns
        -------
        decoder : SoftOutputsBpLsdDecoder
            Cached or newly created decoder.
        """
        # Generate hash key for this configuration
        hash_key = self._hash_matrix_and_priors(H_window, p_window)

        if hash_key in self._decoder_cache:
            return self._decoder_cache[hash_key]

        # Create new decoder
        window_decoder = SoftOutputsBpLsdDecoder(
            H=H_window,
            p=p_window,
            obs_matrix=None,
            **self._bplsd_kwargs,
        )

        # Cache the decoder
        self._decoder_cache[hash_key] = window_decoder
        return window_decoder

    def clear_caches(self) -> None:
        """
        Clear all caches for window structures and decoders.

        This can be useful to free memory or when the decoder configuration changes.
        """
        self._window_structure_cache.clear()
        self._decoder_cache.clear()
        # Clear fixed-class decoder caches
        if hasattr(self, "_H_obs_appended_cache"):
            del self._H_obs_appended_cache
        if hasattr(self, "_decoder_for_fixed_class_cache"):
            del self._decoder_for_fixed_class_cache

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cache usage.

        Returns
        -------
        cache_info : dict
            Dictionary containing cache sizes:
            - window_structures: number of cached window structures
            - decoders: number of cached decoders
        """
        return {
            "window_structures": len(self._window_structure_cache),
            "decoders": len(self._decoder_cache),
        }

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        include_cluster_stats: bool = True,
        compute_logical_gap_proxy: bool = False,
        logical_gap_proxy_method: str | None = None,
        num_classes_to_explore: int | None = None,
        compute_all_intermediate_gap_proxies: bool = False,
        logical_error_distribution: np.ndarray | None = None,
        coverage_fraction: float | None = None,
        num_procs_for_gap: int = 1,
        return_explored_classes: bool = False,
        parallel_verbose: int = 0,
        verbose: bool = False,
        _benchmarking: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.
        include_cluster_stats : bool
            Whether to compute soft outputs related to cluster statistics.
            Defaults to True.
            Automatically set to False when compute_logical_gap_proxy is True.
        compute_logical_gap_proxy : bool
            Whether to compute logical gap proxy. Defaults to False.
        logical_gap_proxy_method : str or None, optional
            Method for exploring logical classes when computing gap proxy:
            - None: Explore all possible logical classes (exact gap proxy).
            - 'nearby': Only explore nearby logical classes (flip one bit at a time).
            - 'random': Randomly sample logical classes uniformly.
            - 'most-likely-first': Deterministically select top classes by distribution.
            - 'weighted-random': Sample classes with probabilities from distribution.
            - 'most-likely-first-adaptive': Like most-likely-first but updates base
              class when better class found during exploration.
            - 'weighted-random-adaptive': Like weighted-random but updates base
              class when better class found during exploration.
            Only used when compute_logical_gap_proxy is True. Defaults to None.
        num_classes_to_explore : int, optional
            Total number of logical classes to explore including the initial best class.
            Required when `logical_gap_proxy_method` is 'random', 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive'.
            Only used when compute_logical_gap_proxy is True. Defaults to None.
        compute_all_intermediate_gap_proxies : bool, optional
            If True and `logical_gap_proxy_method` is 'random', 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive',
            compute additional gap proxies `gap_proxy_{i}` for all i from 2 up to the
            explored number of logical classes. Only used when compute_logical_gap_proxy
            is True. Defaults to False.
        logical_error_distribution : 1D numpy array of float, optional
            Distribution over logical errors with shape (2^k,) where k is the number
            of observables. Index i corresponds to logical error with bit pattern
            i = sum(b_j * 2^j) for j=0..k-1. Higher values indicate more probable
            errors. Required when `logical_gap_proxy_method` is 'most-likely-first',
            'weighted-random', 'most-likely-first-adaptive', or 'weighted-random-adaptive'.
            Also required when `logical_gap_proxy_method` is 'random' and
            `coverage_fraction` is specified and < 1.0.
            Values are auto-normalized internally. Only used when compute_logical_gap_proxy
            is True. Defaults to None.
        coverage_fraction : float, optional
            Fraction of cumulative probability mass to include when sampling
            logical classes for the 'random' gap proxy method. When specified
            (and < 1.0), only logical errors whose cumulative probability (sorted
            by likelihood) is <= coverage_fraction are eligible for uniform sampling.
            Must be in (0, 1]. If 1.0 or None, samples uniformly from all classes
            (default behavior). Requires `logical_error_distribution` when < 1.0.
            Only used when `logical_gap_proxy_method` is 'random'.
            Defaults to None.
        num_procs_for_gap : int, optional
            Number of parallel processes for decoding logical classes. Only effective
            for methods where classes are determined upfront: None (exhaustive),
            'random', 'most-likely-first', 'weighted-random'. Set to -1 to use
            all available CPUs. Raises ValueError for 'nearby' and adaptive methods
            when > 1. Only used when compute_logical_gap_proxy is True.
            Defaults to 1 (sequential).
        return_explored_classes : bool, optional
            If True and compute_logical_gap_proxy is True, include the explored_classes
            dictionary in soft_outputs. This dictionary maps logical class tuples to
            (pred_llr, pred_pattern) tuples for all explored classes, enabling detailed
            analysis of the decoding results across all logical classes.
            Only used when compute_logical_gap_proxy is True. Defaults to False.
        parallel_verbose : int, optional
            Verbosity level for joblib parallel execution. 0 means silent, higher
            values show progress information. Passed directly to joblib.Parallel.
            Only used when compute_logical_gap_proxy is True and num_procs_for_gap != 1.
            Defaults to 0.
        verbose : bool, optional
            If True, print progress information. Defaults to False.
        _benchmarking : bool
            If True, measure elapsed time for each step and print the outcomes in real time.
            Defaults to False.

        Returns
        -------
        pred : np.ndarray
            Predicted error pattern.
        pred_bp : np.ndarray
            Predicted error pattern from BP. It is valid only if the BP is converged.
        converge : bool
            Whether the BP is converged.
        soft_outputs: Dict[str, Any]
            Soft outputs:
            - pred_llr (float): LLR of the predicted error pattern
            - detector_density (float): Fraction of violated detector outcomes
            - clusters (1D numpy array of int): Cluster assignments of each bit (0 = outside clusters)
            - cluster_sizes (1D numpy array of int): Sizes of clusters and the remaining
            region (cluster_sizes[-1])
            - cluster_llrs (1D numpy array of float): LLRs of clusters and the remaining
            region (cluster_llrs[-1])
            - gap_proxy (float): Logical gap proxy (only if compute_logical_gap_proxy=True)
            - gap_proxy_{i} (float): Logical gap proxy after exploring i logical classes
              (only if compute_all_intermediate_gap_proxies=True and logical_gap_proxy_method
              is 'random', 'most-likely-first', 'weighted-random', 'most-likely-first-adaptive',
              or 'weighted-random-adaptive')
            - explored_classes (dict): Dictionary mapping logical class tuples to
              (pred_llr, pred_pattern) tuples (only if return_explored_classes=True
              and compute_logical_gap_proxy=True)
            - initial_logical_class (1D numpy array of bool): Logical class from the initial
              BP+LSD decoding, before any gap proxy exploration updates. Useful for computing
              logical errors relative to the original prediction (only if obs_matrix is provided)
            - cluster_size_norm_frac_{order} (float): Norm fraction of cluster sizes for each order
            - cluster_llr_norm_frac_{order} (float): Norm fraction of cluster LLRs for each order
        """
        if verbose:
            print("Starting BP+LSD decoding...")

        if _benchmarking:
            start_time = time.time()
            step_start = time.time()

        # Gap proxy computation disables cluster stats for efficiency
        if compute_logical_gap_proxy:
            include_cluster_stats = False

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if _benchmarking:
            print(f"[Benchmarking] Input processing: {time.time() - step_start:.6f}s")
            step_start = time.time()

        if verbose:
            print(f"Detector outcomes shape: {detector_outcomes.shape}")
            print(f"Number of violated detectors: {detector_outcomes.sum()}")

        bplsd = self._bplsd
        pred = bplsd.decode(detector_outcomes)
        pred: np.ndarray = pred.astype(bool)
        pred_bp: np.ndarray = bplsd.bp_decoding.astype(bool)

        if _benchmarking:
            print(f"[Benchmarking] BP+LSD decoding: {time.time() - step_start:.6f}s")
            step_start = time.time()

        if verbose:
            print("BP+LSD decoding completed")
            print(f"Predicted error weight: {pred.sum()}")

        ## Soft information
        stats: Dict[str, Any] = bplsd.statistics
        soft_outputs: Dict[str, float | int] = {}

        # Convergence
        converge = bplsd.converge
        if verbose:
            print(f"BP convergence: {converge}")

        if _benchmarking:
            soft_info_start = time.time()

        # LLRs
        llrs = self.bit_llrs
        # bp_llrs = np.array(stats["bit_llrs"])
        # bp_llrs_plus = np.clip(bp_llrs, 0.0, None)

        if _benchmarking:
            print(
                f"[Benchmarking] LLRs extraction: {time.time() - soft_info_start:.6f}s"
            )
            soft_info_start = time.time()

        # Prediction LLR
        soft_outputs["pred_llr"] = float(np.sum(llrs[pred]))
        # soft_outputs["pred_bp_llr"] = float(np.sum(bp_llrs[pred]))

        # Detector density
        soft_outputs["detector_density"] = detector_outcomes.sum() / len(
            detector_outcomes
        )

        if _benchmarking:
            print(
                f"[Benchmarking] Basic soft outputs (pred_llr, detector_density): {time.time() - soft_info_start:.6f}s"
            )
            step_start = time.time()

        if verbose:
            print(f"Prediction LLR: {soft_outputs['pred_llr']:.4f}")
            print(f"Detector density: {soft_outputs['detector_density']:.4f}")

        if include_cluster_stats:
            if verbose:
                print("Computing cluster statistics...")

            if _benchmarking:
                cluster_start = time.time()

            # Build cluster assignments
            individual_cluster_stats_dict: Dict[int, Dict[str, Any]] = stats[
                "individual_cluster_stats"
            ]
            clusters = np.zeros(self.H.shape[1], dtype=np.int_)
            cluster_id = 1
            for data in individual_cluster_stats_dict.values():
                if data.get("active", False):  # Assuming "active" key exists
                    final_bits = data["final_bits"]
                    clusters[final_bits] = cluster_id
                    cluster_id += 1

            if _benchmarking:
                print(
                    f"[Benchmarking] Build cluster assignments: {time.time() - cluster_start:.6f}s"
                )
                cluster_start = time.time()

            # Calculate cluster statistics
            cluster_sizes, cluster_llrs = compute_cluster_stats(clusters, llrs)

            if _benchmarking:
                print(
                    f"[Benchmarking] Compute cluster statistics: {time.time() - cluster_start:.6f}s"
                )
                cluster_start = time.time()

            soft_outputs["clusters"] = clusters  # 1D array of int
            soft_outputs["cluster_sizes"] = cluster_sizes
            soft_outputs["cluster_llrs"] = cluster_llrs

            if _benchmarking:
                print(
                    f"[Benchmarking] Store cluster outputs: {time.time() - cluster_start:.6f}s"
                )

            if _benchmarking:
                step_start = time.time()

            if verbose:
                print(f"Number of active clusters: {cluster_id - 1}")
                print(f"Cluster sizes: {cluster_sizes}")

        # Store initial logical class before gap proxy computation might update pred
        if self.obs_matrix is not None:
            initial_logical_class = (
                (pred.astype("uint8") @ self.obs_matrix.T) % 2
            ).astype(bool)
            # Flatten if needed (sparse matrix returns 2D)
            if hasattr(initial_logical_class, "A1"):
                initial_logical_class = initial_logical_class.A1
            soft_outputs["initial_logical_class"] = initial_logical_class

        if compute_logical_gap_proxy:
            if verbose:
                print("Computing logical gap proxy...")

            if _benchmarking:
                gap_start = time.time()

            (
                gap_proxy,
                best_pred,
                best_pred_llr,
                gap_proxies_by_num_classes,
                explored_classes_result,
            ) = self._compute_logical_gap_proxy(
                detector_outcomes,
                pred,
                soft_outputs["pred_llr"],
                logical_gap_proxy_method=logical_gap_proxy_method,
                num_classes_to_explore=num_classes_to_explore,
                compute_all_intermediate_gap_proxies=compute_all_intermediate_gap_proxies,
                logical_error_distribution=logical_error_distribution,
                coverage_fraction=coverage_fraction,
                num_procs_for_gap=num_procs_for_gap,
                return_explored_classes=return_explored_classes,
                parallel_verbose=parallel_verbose,
                verbose=verbose,
            )

            if _benchmarking:
                print(
                    f"[Benchmarking] Logical gap proxy computation: {time.time() - gap_start:.6f}s"
                )
                gap_start = time.time()

            soft_outputs["gap_proxy"] = gap_proxy
            if compute_all_intermediate_gap_proxies and gap_proxies_by_num_classes:
                for num_classes, gap in sorted(gap_proxies_by_num_classes.items()):
                    soft_outputs[f"gap_proxy_{num_classes}"] = gap
            if explored_classes_result is not None:
                soft_outputs["explored_classes"] = explored_classes_result

            # Update prediction and related soft outputs if a better one was found
            if best_pred_llr < soft_outputs["pred_llr"]:
                if verbose:
                    print(
                        f"  Updating prediction: {soft_outputs['pred_llr']:.4f} -> {best_pred_llr:.4f}"
                    )
                pred = best_pred
                soft_outputs["pred_llr"] = best_pred_llr

            if _benchmarking:
                print(
                    f"[Benchmarking] Update prediction from gap proxy: {time.time() - gap_start:.6f}s"
                )
                step_start = time.time()

        if verbose:
            print("BP+LSD decoding process completed!")

        if _benchmarking:
            print(f"[Benchmarking] Total decode time: {time.time() - start_time:.6f}s")

        return pred, pred_bp, converge, soft_outputs

    def decode_sliding_window(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        window_size: int,
        commit_size: int,
        verbose: bool = False,
        _benchmarking: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode detector outcomes using (window_size, commit_size)-sliding window method.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.
        window_size : int
            Number of rounds in each window.
        commit_size : int
            Number of rounds for each commitment.
        verbose : bool, optional
            If True, print progress information. Defaults to False.
        _benchmarking : bool
            If True, measure elapsed time for each step and print the outcomes in real time.
            Defaults to False.

        Returns
        -------
        pred : 1D numpy array of bool
            Predicted error pattern.
        soft_outputs : dict
            Aggregated soft outputs from all windows containing:
            - all_clusters: list of cluster assignments for each window
            - committed_clusters: list of boolean arrays after each window.
            True if fault is committed and in a cluster, False otherwise.
            - committed_faults: list of boolean arrays after each window.
            True if fault is committed, False otherwise.
        """

        self._check_detector_time_coords_validity()

        if window_size <= commit_size:
            raise ValueError("W must be greater than F")

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if verbose:
            print(
                f"Starting sliding window decoding with W={window_size}, F={commit_size}"
            )

        if _benchmarking:
            start_time = time.time()
            step_start = time.time()
            # Dictionary to track benchmarking times for each step
            benchmark_times = {
                "extract_detector_mask": [],
                "extract_h_matrix_rows": [],
                "find_active_faults": [],
                "extract_submatrices": [],
                "create_decoder": [],
                "decode": [],
                "convert_to_full_size": [],
                "determine_commits": [],
                "update_masks": [],
                "update_detectors_prediction": [],
                "window_total": [],
            }

        # Initialize prediction array
        pred = np.zeros(self.H.shape[1], dtype=bool)

        # Get detector time coordinates
        detector_times = self.detector_time_coords
        max_time = detector_times.max()

        # Storage for aggregated soft outputs
        window_clusters = []

        # Storage for window-wise committed clusters (boolean: True if committed AND in cluster)
        window_committed_clusters = []

        # Storage for window-wise committed faults (boolean: True if committed)
        window_committed_faults = []

        if _benchmarking:
            print(f"[Benchmarking] Initialization: {time.time() - step_start:.6f}s")
            step_start = time.time()

        if verbose:
            print(f"Max detector time: {max_time}")
            print(f"Total detectors: {len(detector_times)}")

        w = 0
        while True:
            if _benchmarking:
                window_start_time = time.time()

            window_start = w * commit_size
            window_end = w * commit_size + window_size - 1

            if verbose:
                print(f"\nWindow {w}: time range [{window_start}, {window_end}]")

            # Check if this is the final window
            is_final_window = window_end >= max_time

            if _benchmarking:
                step_time = time.time()

            # Get cached window structure or compute it
            window_structure = self._get_or_create_window_structure(
                window_size, commit_size, w
            )
            window_detector_mask = window_structure["window_detector_mask"]
            H_window = window_structure["H_window_base"]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_detector_mask"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Get window structure (cached): {elapsed:.6f}s"
                )
                step_time = time.time()

            if not np.any(window_detector_mask):
                if verbose:
                    print(f"No detectors in window {w}, stopping")
                break

            # Extract detector outcomes for this window
            det_outcomes_window = detector_outcomes[window_detector_mask]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_h_matrix_rows"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Extract detector outcomes: {elapsed:.6f}s"
                )
                step_time = time.time()

            # Find columns (faults) that have at least one nonzero element
            # Exclude already-committed faults from this window
            fault_mask = np.asarray(H_window.sum(axis=0) > 0).ravel()
            if window_committed_faults:
                # Compute committed faults mask from previous windows
                committed_faults_mask = np.any(window_committed_faults, axis=0)
                fault_mask = fault_mask & ~committed_faults_mask

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["find_active_faults"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Find active faults: {elapsed:.6f}s")
                step_time = time.time()

            if not np.any(fault_mask):
                if verbose:
                    print(f"No active faults in window {w}, skipping")
                w += 1
                continue

            # Extract submatrices
            H_window = H_window[:, fault_mask]
            p_window = self.priors[fault_mask]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_submatrices"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Extract submatrices: {elapsed:.6f}s"
                )
                step_time = time.time()

            if verbose:
                print(f"Window matrix shape: {H_window.shape}")
                print(f"Active faults: {fault_mask.sum()}")
                print(f"Violated detectors: {det_outcomes_window.sum()}")

            # Get cached decoder or create new one
            window_decoder = self._get_or_create_window_decoder(H_window, p_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["create_decoder"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Get/create decoder (cached): {elapsed:.6f}s"
                )
                step_time = time.time()

            # Decode window
            pred_window_small, _, _, soft_outputs_window = window_decoder.decode(
                det_outcomes_window,
                include_cluster_stats=True,
                compute_logical_gap_proxy=False,
                verbose=False,
            )

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["decode"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Decode: {elapsed:.6f}s")
                step_time = time.time()

            # Convert window prediction to full size
            pred_window = np.zeros(self.H.shape[1], dtype=bool)
            pred_window[fault_mask] = pred_window_small

            # Convert clusters to full size
            clusters_window = np.zeros(self.H.shape[1], dtype=int)
            clusters_window[fault_mask] = soft_outputs_window["clusters"]

            # Store window soft outputs
            window_clusters.append(clusters_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["convert_to_full_size"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Convert to full size: {elapsed:.6f}s"
                )
                step_time = time.time()

            # Determine which faults to commit
            if is_final_window:
                # Final window: commit all faults
                pred_to_commit = pred_window
                commit_mask = fault_mask
                if verbose:
                    print("Final window: committing all faults")
            else:
                # Regular window: commit only faults involved in detectors within [w*F, w*F+F-1]
                commit_start = w * commit_size
                commit_end = w * commit_size + commit_size - 1
                commit_detector_mask = (detector_times >= commit_start) & (
                    detector_times <= commit_end
                )

                if np.any(commit_detector_mask):
                    # Find faults involved in commit region detectors
                    H_commit_rows = self.H[commit_detector_mask, :]
                    commit_mask = np.asarray(H_commit_rows.sum(axis=0) > 0).ravel()
                    # Exclude already committed faults from commit region
                    if window_committed_faults:
                        committed_faults_mask = np.any(window_committed_faults, axis=0)
                        commit_mask &= ~committed_faults_mask

                    pred_to_commit = pred_window.copy()
                    pred_to_commit[~commit_mask] = False
                else:
                    pred_to_commit = np.zeros(self.H.shape[1], dtype=bool)
                    commit_mask = np.zeros(self.H.shape[1], dtype=bool)

                if verbose:
                    print(f"Commit region: [{commit_start}, {commit_end}]")
                    print(f"Committing {pred_to_commit.sum()} faults")

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["determine_commits"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Determine commits: {elapsed:.6f}s")
                step_time = time.time()

            # No need to maintain separate committed faults mask - computed from window_committed_faults when needed

            # Track committed clusters (boolean: True if committed AND in cluster)
            committed_clusters_current_window = commit_mask & (clusters_window > 0)

            # Track committed faults (boolean: True if committed)
            committed_faults_current_window = commit_mask.copy()

            # Store both arrays
            window_committed_clusters.append(committed_clusters_current_window)
            window_committed_faults.append(committed_faults_current_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["update_masks"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Update masks: {elapsed:.6f}s")
                step_time = time.time()

            # Update detector outcomes and prediction
            detector_update = ((pred_to_commit.astype(np.uint8) @ self.H.T) % 2).astype(
                bool
            )
            detector_outcomes ^= detector_update
            pred ^= pred_to_commit

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["update_detectors_prediction"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Update detectors/prediction: {elapsed:.6f}s"
                )

            if verbose:
                print(f"Updated {detector_update.sum()} detector outcomes")
                print(f"Total prediction weight: {pred.sum()}")
                print(f"Remaining violated detectors: {detector_outcomes.sum()}")

            if _benchmarking:
                window_elapsed = time.time() - window_start_time
                benchmark_times["window_total"].append(window_elapsed)
                print(f"[Benchmarking] Window {w} total: {window_elapsed:.6f}s")

            # Break if final window
            if is_final_window:
                break

            w += 1

        # Create aggregated soft outputs
        soft_outputs = {
            "all_clusters": window_clusters,
            "committed_clusters": window_committed_clusters,
            "committed_faults": window_committed_faults,
        }

        if verbose:
            print(f"\nSliding window decoding completed!")
            print(f"Total windows processed: {len(window_clusters)}")
            print(f"Final prediction weight: {pred.sum()}")

        if _benchmarking:
            total_time = time.time() - start_time
            print(f"[Benchmarking] Total sliding window decode time: {total_time:.6f}s")

            # Print aggregated benchmark summary
            print("\n[Benchmarking] ========== SUMMARY ACROSS ALL WINDOWS ==========")
            print(f"Total windows processed: {len(benchmark_times['window_total'])}")
            print("\nStep-by-step breakdown (mean ± std) [total]:\n")

            for step_name, times in benchmark_times.items():
                if times:  # Only show steps that were actually executed
                    times_array = np.array(times)
                    mean_time = np.mean(times_array)
                    std_time = np.std(times_array)
                    total_step_time = np.sum(times_array)
                    print(
                        f"  {step_name:<30}: {mean_time:.6f} ± {std_time:.6f}s [total: {total_step_time:.6f}s]"
                    )

            print("\n[Benchmarking] ===============================================")

        return pred, soft_outputs

    def simulate_single(self, sliding_window=False, seed=None, **kwargs):
        if seed is not None:
            rng = np.random.default_rng(seed)
            errors = rng.random(self.H.shape[1], dtype=np.float64) < self.priors
        else:
            errors = np.random.random(self.H.shape[1]) < self.priors

        det_outcomes = (errors @ self.H.T % 2).astype(bool)

        if sliding_window:
            pred, soft_outputs = self.decode_sliding_window(det_outcomes, **kwargs)
        else:
            pred, _, _, soft_outputs = self.decode(det_outcomes, **kwargs)

        residual = pred ^ errors
        valid = not bool(np.any(residual @ self.H.T % 2))
        if not valid:
            raise ValueError("Decoding outcome invalid")

        fail = bool(np.any((residual @ self.obs_matrix.T) % 2))

        return fail, soft_outputs
