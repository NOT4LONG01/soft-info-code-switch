import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)


if __name__ == "__main__":
    # Select which datasets to collect
    collect_cluster_stats = False
    collect_gap_proxy = True

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

    # Gap proxy configuration
    ascending_confidences_gap_proxy = {
        "gap_proxy": True,
    }
    gap_proxy_data_dir_name = "bb_minsum_iter30_lsd0_raw_gap_proxy_nearby"
    gap_proxy_dataset_name = "bb_gap_proxy"

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

    # Process gap_proxy data
    if collect_gap_proxy:
        gap_proxy_data_dir = str(DATA_DIR / gap_proxy_data_dir_name)
        if os.path.exists(gap_proxy_data_dir):
            print("\nProcessing BB code gap_proxy data...")
            process_dataset(
                data_dir=gap_proxy_data_dir,
                dataset_name=gap_proxy_dataset_name,
                ascending_confidences=ascending_confidences_gap_proxy,
                dataset_type="bb",
                decimals=2,
                verbose=False,
            )
        else:
            print(f"\nSkipping gap_proxy data: {gap_proxy_data_dir} not found")

    print("\nData collection complete!")
