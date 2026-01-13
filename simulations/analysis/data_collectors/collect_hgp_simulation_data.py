import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)

# Mapping from method name to directory/dataset suffix
GAP_PROXY_METHOD_SUFFIXES = {
    "nearby": "nearby",
    "random": "random",
    "mlf": "mlf",
    "weighted-random": "wr",
    "mlf-adaptive": "mlfa",
    "weighted-random-adaptive": "wra",
}


if __name__ == "__main__":
    # Select which datasets to collect
    collect_cluster_stats = False
    collect_gap_proxy = True

    # Gap proxy configuration
    # Options: 'nearby', 'random', 'mlf', 'weighted-random', 'mlf-adaptive', 'weighted-random-adaptive'
    gap_proxy_method = "nearby"
    num_logical_classes = (
        24  # Number of logical classes explored (not used for 'nearby')
    )
    # For 'random' method only: fraction of cumulative prob mass (None or 1.0 = no restriction)
    coverage_fraction = None
    compute_all_gap_proxies = True  # Whether gap_proxy_2, gap_proxy_3, ... exist

    # Cluster stats configuration
    orders = [0.5, 1, 2, np.inf]
    ascending_confidences = {
        "pred_llr": False,
        "detector_density": False,
        "cluster_size_norm_frac": False,
        "cluster_llr_norm_frac": False,
    }
    data_dir_name = "hgp_minsum_iter30_lsd0_raw"
    dataset_name = "hgp"

    # Process cluster stats data
    if collect_cluster_stats:
        data_dir = str(DATA_DIR / data_dir_name)
        if os.path.exists(data_dir):
            print("Processing HGP code cluster stats data...")
            process_dataset(
                data_dir=data_dir,
                dataset_name=dataset_name,
                dataset_type="hgp",
                ascending_confidences=ascending_confidences,
                orders=orders,
                decimals=(lambda by: 2 if by == "pred_llr" else 4),
                verbose=False,
            )
        else:
            print(f"Skipping cluster stats data: {data_dir} not found")

    # Process gap_proxy data
    if collect_gap_proxy:
        method_suffix = GAP_PROXY_METHOD_SUFFIXES[gap_proxy_method]

        # Build data directory and dataset names
        if gap_proxy_method == "nearby":
            gap_proxy_data_dir_name = (
                f"hgp_minsum_iter30_lsd0_raw_gap_proxy_{method_suffix}"
            )
            gap_proxy_dataset_name = "hgp_gap_proxy"
        else:
            # Build coverage suffix for 'random' method with coverage_fraction
            cov_suffix = ""
            if (
                gap_proxy_method == "random"
                and coverage_fraction is not None
                and coverage_fraction < 1.0
            ):
                cov_suffix = f"_cov{coverage_fraction}"
            gap_proxy_data_dir_name = f"hgp_minsum_iter30_lsd0_raw_gap_proxy_{method_suffix}_{num_logical_classes}{cov_suffix}"
            gap_proxy_dataset_name = (
                f"hgp_gap_proxy_{method_suffix}_{num_logical_classes}{cov_suffix}"
            )

        gap_proxy_data_dir = str(DATA_DIR / gap_proxy_data_dir_name)

        if os.path.exists(gap_proxy_data_dir):
            if gap_proxy_method == "nearby":
                print(f"\nProcessing HGP code gap_proxy ({gap_proxy_method}) data...")
            elif cov_suffix:
                print(
                    f"\nProcessing HGP code gap_proxy ({gap_proxy_method} with "
                    f"{num_logical_classes} classes, coverage={coverage_fraction}) data..."
                )
            else:
                print(
                    f"\nProcessing HGP code gap_proxy ({gap_proxy_method} with "
                    f"{num_logical_classes} classes) data..."
                )

            # Build ascending_confidences dict for gap_proxy columns
            ascending_confidences_gap_proxy = {"gap_proxy": True}

            # Add gap_proxy_i columns if compute_all_gap_proxies is True
            if compute_all_gap_proxies and gap_proxy_method != "nearby":
                for i in range(2, num_logical_classes + 1):
                    ascending_confidences_gap_proxy[f"gap_proxy_{i}"] = True

            process_dataset(
                data_dir=gap_proxy_data_dir,
                dataset_name=gap_proxy_dataset_name,
                ascending_confidences=ascending_confidences_gap_proxy,
                dataset_type="hgp",
                decimals=2,
                verbose=False,
            )
        else:
            print(
                f"\nSkipping gap_proxy ({gap_proxy_method}) data: "
                f"{gap_proxy_data_dir} not found"
            )

    print("\nData collection complete!")
