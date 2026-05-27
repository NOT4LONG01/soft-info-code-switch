import os
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np
from scipy import sparse

from soft_info.decoders.legacy import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    get_existing_shots,
    bplsd_simulation_task_parallel,
    precompute_logical_error_distribution,
)
from simulations.utils.simulation_utils_legacy import (
    bplsd_simulation_task_parallel_legacy,
)
from simulations.utils.build_circuit import build_BB_circuit, get_BB_distance


def simulate(
    shots: int,
    p: float,
    n: int,
    T: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
    compute_logical_gap_proxy: bool = False,
    logical_gap_proxy_method: str | None = None,
    num_classes_to_explore: int | None = None,
    compute_all_intermediate_gap_proxies: bool = False,
    logical_error_distribution: np.ndarray | None = None,
    coverage_fraction: float | None = None,
    precompute_distribution_shots: int | None = None,
    include_cluster_stats: bool = True,
) -> None:
    """
    Run the simulation for a given (p, n, T) configuration, saving results in batches.
    Results include a Feather file for scalar data and NumPy files for ragged arrays.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    T : int
        Number of rounds.
    data_dir : str
        Base directory to store output subdirectories.
    n_jobs : int
        Number of parallel jobs for `task_parallel`.
    repeat : int
        Number of repeats for `task_parallel`.
    shots_per_batch : int
        Number of shots to simulate and save per batch file.
    decoder_prms : Dict[str, Any], optional
        Parameters for the SoftOutputsBpLsdDecoder.
    compute_logical_gap_proxy : bool, optional
        Whether to compute logical gap proxy. Defaults to False.
    logical_gap_proxy_method : str or None, optional
        Method for exploring logical classes when computing gap proxy:
        - None: Explore all possible logical classes (exact gap proxy).
        - 'nearby': Only explore nearby logical classes (flip one bit at a time).
        - 'random': Randomly sample logical classes uniformly.
        - 'most-likely-first': Deterministically select top classes by distribution.
        - 'weighted-random': Sample classes with probabilities from distribution.
        Only used when compute_logical_gap_proxy is True. Defaults to None.
    num_classes_to_explore : int, optional
        Total number of logical classes to explore including the initial best class.
        Required when `logical_gap_proxy_method` is 'random', 'most-likely-first',
        or 'weighted-random'. Only used when compute_logical_gap_proxy is True.
        Defaults to None.
    compute_all_intermediate_gap_proxies : bool, optional
        If True and `logical_gap_proxy_method` is 'random', 'most-likely-first',
        or 'weighted-random', compute additional gap proxies `gap_proxy_{i}` for
        all i from 2 up to the explored number of logical classes. Only used when
        compute_logical_gap_proxy is True. Defaults to False.
    logical_error_distribution : 1D numpy array of float, optional
        Pre-computed logical error distribution for 'most-likely-first' or
        'weighted-random' methods. Also required when `logical_gap_proxy_method`
        is 'random' and `coverage_fraction` is specified and < 1.0.
        If None and method requires it, will be computed automatically using
        `precompute_distribution_shots`. Defaults to None.
    coverage_fraction : float, optional
        Fraction of cumulative probability mass to include when sampling
        logical classes for the 'random' gap proxy method. When specified
        (and < 1.0), only logical errors whose cumulative probability (sorted
        by likelihood) is <= coverage_fraction are eligible for uniform sampling.
        Must be in (0, 1]. If 1.0 or None, samples uniformly from all classes.
        Requires `logical_error_distribution` when < 1.0.
        Only used when `logical_gap_proxy_method` is 'random'. Defaults to None.
    precompute_distribution_shots : int, optional
        Number of shots to use for pre-computing logical error distribution when
        `logical_gap_proxy_method` is 'most-likely-first' or 'weighted-random'
        or when 'random' with `coverage_fraction` < 1.0
        and no distribution is provided. Defaults to None (uses 100,000 shots).
    include_cluster_stats : bool, optional
        Whether to include cluster statistics. Defaults to True.

    Returns
    -------
    None
        This function writes results to files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_dirname = f"n{n}_T{T}_p{p}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, T={T} into {sub_dirname}"
    )

    # Create the circuit once for this (p, n, T) configuration
    circuit = build_BB_circuit(p=p, n=n, T=T)

    # Save prior probabilities if not exists
    prior_path = os.path.join(sub_data_dir, "priors.npy")
    if not os.path.exists(prior_path):
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit)
        np.save(prior_path, decoder.priors)

    # Pre-compute logical error distribution for methods that require it
    requires_distribution = logical_gap_proxy_method in (
        "most-likely-first",
        "weighted-random",
        "most-likely-first-adaptive",
        "weighted-random-adaptive",
    ) or (
        logical_gap_proxy_method == "random"
        and coverage_fraction is not None
        and coverage_fraction < 1.0
    )
    if requires_distribution and logical_error_distribution is None:
        # Store distribution in shared location (independent of gap proxy method)
        # Pattern: data/logical_error_distributions/bb_{decoder_settings}/{code_params}.npy
        base_data_dir = os.path.dirname(data_dir)
        bp_method_short = (
            "minsum"
            if decoder_prms["bp_method"] == "minimum_sum"
            else decoder_prms["bp_method"]
        )
        lsd_method_short = decoder_prms["lsd_method"].lower().replace("_", "")
        decoder_settings = (
            f"{bp_method_short}_iter{decoder_prms['max_iter']}_{lsd_method_short}"
        )
        distribution_dir = os.path.join(
            base_data_dir, f"logical_error_distributions/bb_{decoder_settings}"
        )
        os.makedirs(distribution_dir, exist_ok=True)
        distribution_path = os.path.join(distribution_dir, f"n{n}_T{T}_p{p}.npy")

        if precompute_distribution_shots is None:
            precompute_distribution_shots = 100_000

        logical_error_distribution = precompute_logical_error_distribution(
            circuit=circuit,
            save_path=distribution_path,
            shots=precompute_distribution_shots,
            n_jobs=n_jobs,
            decoder_params=decoder_prms,
        )

    # Determine the next file index
    next_idx = (
        max([info[0] for info in existing_files_info], default=0) + 1
        if existing_files_info
        else 1
    )

    # Simulate and save in batches
    current_simulated_for_config = 0
    while remaining > 0:
        to_run = min(shots_per_batch, remaining)
        if to_run == 0:
            break  # Should not happen if remaining > 0

        # Define the specific directory for this batch's output
        batch_output_dir = os.path.join(sub_data_dir, f"batch_{next_idx}_{to_run}")

        t0_batch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, T={T}. Output to: {batch_output_dir}"
        )

        df_new, clusters_csr, preds_csr, preds_bp_csr = bplsd_simulation_task_parallel(
            shots=to_run,
            circuit=circuit,  # Pass the pre-built circuit
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
            compute_logical_gap_proxy=compute_logical_gap_proxy,
            logical_gap_proxy_method=logical_gap_proxy_method,
            num_classes_to_explore=num_classes_to_explore,
            compute_all_intermediate_gap_proxies=compute_all_intermediate_gap_proxies,
            logical_error_distribution=logical_error_distribution,
            coverage_fraction=coverage_fraction,
            include_cluster_stats=include_cluster_stats,
        )

        # Prepare filenames for this batch (now with fixed names within batch_output_dir)
        fp_feather = os.path.join(batch_output_dir, "scalars.feather")
        fp_clusters = os.path.join(batch_output_dir, "clusters.npz")
        fp_preds = os.path.join(batch_output_dir, "preds.npz")
        fp_preds_bp = os.path.join(batch_output_dir, "preds_bp.npz")

        # Convert dtypes and save
        os.makedirs(batch_output_dir, exist_ok=True)
        df_new = _convert_df_dtypes_for_feather(
            df_new.copy()
        )  # Use .copy() to avoid SettingWithCopyWarning
        df_new.to_feather(fp_feather)

        # Save sparse matrices as compressed NPZ files
        if clusters_csr is not None:
            sparse.save_npz(fp_clusters, clusters_csr)
        sparse.save_npz(fp_preds, preds_csr)
        sparse.save_npz(fp_preds_bp, preds_bp_csr)

        current_simulated_for_config += to_run
        remaining -= to_run
        total_processed_for_config = current_simulated_for_config  # Corrected logic

        print(
            f"   Created files in {batch_output_dir} with {to_run} shots. "
            f"{total_processed_for_config} / {shots - total_existing} new shots processed for this config. "
            f"{remaining} shots still remaining for this config."
        )
        next_idx += 1


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore", message="A worker stopped while some jobs were given to the executor."
    )

    # plist = np.arange(1e-3, 4.1e-3, 1e-3).round(4)
    # nlist = [72, 144]  # [72, 108, 144, 288]

    plist = [0.003]
    nlist = [144]

    shots_per_batch = round(1e4)
    total_shots = round(1e4)
    compute_logical_gap_proxy = True
    include_cluster_stats = False
    logical_gap_proxy_method = (
        "random"  # None, 'nearby', 'random', 'most-likely-first', or 'weighted-random'
    )
    num_classes_to_explore = 24  # Required when method is 'random', 'most-likely-first', or 'weighted-random'
    coverage_fraction = None  # For 'random' method: fraction of cumulative prob mass to sample from (e.g., 0.3)
    compute_all_intermediate_gap_proxies = True
    precompute_distribution_shots = 1000 * 2**12

    n_jobs = 126
    repeat = 1
    if compute_logical_gap_proxy:
        if logical_gap_proxy_method == "nearby":
            dir_name = "bb_minsum_iter30_lsd0_raw_gap_proxy_nearby"
        elif logical_gap_proxy_method == "random":
            assert num_classes_to_explore is not None
            if coverage_fraction is not None and coverage_fraction < 1.0:
                dir_name = f"bb_minsum_iter30_lsd0_raw_gap_proxy_random_{num_classes_to_explore}_cov{coverage_fraction}"
            else:
                dir_name = f"bb_minsum_iter30_lsd0_raw_gap_proxy_random_{num_classes_to_explore}"
        elif logical_gap_proxy_method == "most-likely-first":
            assert num_classes_to_explore is not None
            dir_name = (
                f"bb_minsum_iter30_lsd0_raw_gap_proxy_mlf_{num_classes_to_explore}"
            )
        elif logical_gap_proxy_method == "weighted-random":
            assert num_classes_to_explore is not None
            dir_name = (
                f"bb_minsum_iter30_lsd0_raw_gap_proxy_wr_{num_classes_to_explore}"
            )
        elif logical_gap_proxy_method == "most-likely-first-adaptive":
            assert num_classes_to_explore is not None
            dir_name = (
                f"bb_minsum_iter30_lsd0_raw_gap_proxy_mlfa_{num_classes_to_explore}"
            )
        elif logical_gap_proxy_method == "weighted-random-adaptive":
            assert num_classes_to_explore is not None
            dir_name = (
                f"bb_minsum_iter30_lsd0_raw_gap_proxy_wra_{num_classes_to_explore}"
            )
        else:
            dir_name = "bb_minsum_iter30_lsd0_raw_gap_proxy"
    else:
        dir_name = "bb_minsum_iter30_lsd0_raw"

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, f"data/{dir_name}")
    os.makedirs(data_dir, exist_ok=True)

    print("nlist =", nlist)
    print("plist =", plist)
    print("decoder_prms =", decoder_prms)
    print("compute_logical_gap_proxy =", compute_logical_gap_proxy)
    print("logical_gap_proxy_method =", logical_gap_proxy_method)
    print("num_classes_to_explore =", num_classes_to_explore)
    if logical_gap_proxy_method == "random" and coverage_fraction is not None:
        print("coverage_fraction =", coverage_fraction)
    if logical_gap_proxy_method in (
        "most-likely-first",
        "weighted-random",
        "most-likely-first-adaptive",
        "weighted-random-adaptive",
    ) or (
        logical_gap_proxy_method == "random"
        and coverage_fraction is not None
        and coverage_fraction < 1.0
    ):
        print("precompute_distribution_shots =", precompute_distribution_shots)

    print(f"\n==== Starting simulations up to {total_shots} shots ====")
    for n in nlist:
        T = get_BB_distance(n)
        for i_p, p in enumerate(plist):
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                T=T,
                data_dir=data_dir,
                n_jobs=n_jobs,
                repeat=repeat,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
                compute_logical_gap_proxy=compute_logical_gap_proxy,
                include_cluster_stats=include_cluster_stats,
                logical_gap_proxy_method=logical_gap_proxy_method,
                num_classes_to_explore=num_classes_to_explore,
                coverage_fraction=coverage_fraction,
                compute_all_intermediate_gap_proxies=compute_all_intermediate_gap_proxies,
                precompute_distribution_shots=precompute_distribution_shots,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
