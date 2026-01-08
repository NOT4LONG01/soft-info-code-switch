import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)


if __name__ == "__main__":
    # Select which datasets to collect
    collect_cluster_stats = False
    collect_gap_proxy_nearby = False
    collect_gap_proxy_random = False
    collect_gap_proxy_mlf = True  # most-likely-first method

    # Gap proxy random configuration
    explore_random_logical_classes = 24  # Number of random logical classes explored
    compute_all_random_gap_proxies = True  # Whether gap_proxy_2, gap_proxy_3, ... exist

    # Gap proxy most-likely-first configuration
    explore_mlf_logical_classes = 24  # Number of most-likely logical classes explored
    compute_all_mlf_gap_proxies = True  # Whether gap_proxy_2, gap_proxy_3, ... exist

    # orders = [0.5, 1, 2, np.inf]
    orders = [2]
    use_old_format_data = False

    # Cluster stats configuration
    ascending_confidences = {
        # "pred_llr": False,
        # "detector_density": False,
        # "cluster_size_norm_frac": False,
        "cluster_llr_norm_frac": False,
    }
    if use_old_format_data:
        data_dir_name = "bb_minsum_iter30_lsd0"
        dataset_name = "bb_old"
    else:
        data_dir_name = "bb_minsum_iter30_lsd0_raw"
        dataset_name = "bb"

    # Gap proxy (nearby) configuration
    ascending_confidences_gap_proxy_nearby = {
        "gap_proxy": True,
    }
    gap_proxy_nearby_data_dir_name = "bb_minsum_iter30_lsd0_raw_gap_proxy_nearby"
    gap_proxy_nearby_dataset_name = "bb_gap_proxy"

    # Gap proxy (random) configuration
    gap_proxy_random_data_dir_name = (
        f"bb_minsum_iter30_lsd0_raw_gap_proxy_random_{explore_random_logical_classes}"
    )
    gap_proxy_random_dataset_name = (
        f"bb_gap_proxy_random_{explore_random_logical_classes}"
    )

    # Gap proxy (most-likely-first) configuration
    gap_proxy_mlf_data_dir_name = (
        f"bb_minsum_iter30_lsd0_raw_gap_proxy_mlf_{explore_mlf_logical_classes}"
    )
    gap_proxy_mlf_dataset_name = f"bb_gap_proxy_mlf_{explore_mlf_logical_classes}"

    # Process cluster stats data
    if collect_cluster_stats:
        data_dir = str(DATA_DIR / data_dir_name)
        print("Processing BB code cluster stats data...")
        process_dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            dataset_type="bb",
            ascending_confidences=ascending_confidences,
            orders=orders,
            decimals=(lambda by: 2 if by == "pred_llr" else 4),
            verbose=False,
        )

    # Process gap_proxy (nearby) data
    if collect_gap_proxy_nearby:
        gap_proxy_data_dir = str(DATA_DIR / gap_proxy_nearby_data_dir_name)
        if os.path.exists(gap_proxy_data_dir):
            print("\nProcessing BB code gap_proxy (nearby) data...")
            process_dataset(
                data_dir=gap_proxy_data_dir,
                dataset_name=gap_proxy_nearby_dataset_name,
                ascending_confidences=ascending_confidences_gap_proxy_nearby,
                dataset_type="bb",
                decimals=2,
                verbose=False,
            )
        else:
            print(f"\nSkipping gap_proxy (nearby) data: {gap_proxy_data_dir} not found")

    # Process gap_proxy (random) data
    if collect_gap_proxy_random:
        gap_proxy_data_dir = str(DATA_DIR / gap_proxy_random_data_dir_name)
        if os.path.exists(gap_proxy_data_dir):
            print(
                f"\nProcessing BB code gap_proxy (random with "
                f"{explore_random_logical_classes} classes) data..."
            )

            # Build ascending_confidences dict for gap_proxy columns
            ascending_confidences_gap_proxy_random = {
                "gap_proxy": True,
            }

            # Add gap_proxy_i columns if compute_all_random_gap_proxies is True
            if compute_all_random_gap_proxies:
                for i in range(2, explore_random_logical_classes + 1):
                    ascending_confidences_gap_proxy_random[f"gap_proxy_{i}"] = True

            process_dataset(
                data_dir=gap_proxy_data_dir,
                dataset_name=gap_proxy_random_dataset_name,
                ascending_confidences=ascending_confidences_gap_proxy_random,
                dataset_type="bb",
                decimals=2,
                verbose=False,
            )
        else:
            print(f"\nSkipping gap_proxy (random) data: {gap_proxy_data_dir} not found")

    # Process gap_proxy (most-likely-first) data
    if collect_gap_proxy_mlf:
        gap_proxy_data_dir = str(DATA_DIR / gap_proxy_mlf_data_dir_name)
        if os.path.exists(gap_proxy_data_dir):
            print(
                f"\nProcessing BB code gap_proxy (most-likely-first with "
                f"{explore_mlf_logical_classes} classes) data..."
            )

            # Build ascending_confidences dict for gap_proxy columns
            ascending_confidences_gap_proxy_mlf = {
                "gap_proxy": True,
            }

            # Add gap_proxy_i columns if compute_all_mlf_gap_proxies is True
            if compute_all_mlf_gap_proxies:
                for i in range(2, explore_mlf_logical_classes + 1):
                    ascending_confidences_gap_proxy_mlf[f"gap_proxy_{i}"] = True

            process_dataset(
                data_dir=gap_proxy_data_dir,
                dataset_name=gap_proxy_mlf_dataset_name,
                ascending_confidences=ascending_confidences_gap_proxy_mlf,
                dataset_type="bb",
                decimals=2,
                verbose=False,
            )
        else:
            print(
                f"\nSkipping gap_proxy (most-likely-first) data: {gap_proxy_data_dir} not found"
            )

    print("\nData collection complete!")
