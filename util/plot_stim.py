import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse
import sys

# Import dist dicts from the canonical source (src/decoder/codes.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'decoder'))
from codes import GO03_DICT, EQR_ISO_DICT, EQR_SD_CSS_DICT, JA25_DICT

# Maps code_type → distance dict for codes where d is determined by n
_DIST_DICTS = {
    'go03_self_dual':    GO03_DICT,
    'eqr_iso_dual':      EQR_ISO_DICT,
    'eqr_self_dual_css': EQR_SD_CSS_DICT,
    'ja25_transversal_t': JA25_DICT,
}

ALPHA = 0.05

def get_code_metadata(code_type, n, d_from_csv):
    dist_dict = _DIST_DICTS.get(code_type)
    if dist_dict:
        d = dist_dict.get(n, d_from_csv)
        return f"n={n}, d={d}", d
    else:
        # Geometric codes (cubic, tetrahedral, triangular, square): d is the parameter
        return f"d={d_from_csv} (n={n})", d_from_csv

def load_and_aggregate_data(directory, decoder_filter=None):
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return pd.DataFrame()

    if decoder_filter:
        all_files = glob.glob(os.path.join(directory, f"*_{decoder_filter}_rank*.csv"))
    else:
        all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {directory}" + (f" for decoder={decoder_filter}" if decoder_filter else ""))
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} files. Loading and Aggregating...")
    df_list = []
    
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not df_list: return pd.DataFrame()
    
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # Fill defaults
    defaults = {'code_type': 'unknown', 'noise_model': 'unknown', 'n': 0, 'd': 0, 'p': 0.0, 'shots': 0, 'errors': 0, 'variant': 'base'}
    for col, val in defaults.items():
        if col not in raw_df.columns:
            raw_df[col] = val

    # Aggregate
    group_cols = ['code_type', 'noise_model', 'n', 'd', 'p', 'variant']
    agg_rules = {'shots': 'sum', 'errors': 'sum'}
    
    if 'mean_objective_per_syndrome' in raw_df.columns:
        agg_rules['mean_objective_per_syndrome'] = 'mean'
    if 'average_cpu_time_seconds' in raw_df.columns:
        agg_rules['average_cpu_time_seconds'] = 'mean'
        
    agg_df = raw_df.groupby(group_cols, as_index=False).agg(agg_rules)
    agg_df['total_logical_error_rate'] = agg_df['errors'] / agg_df['shots']
    
    print(f"Aggregated {len(raw_df)} raw rows into {len(agg_df)} unique data points.")
    return agg_df

def plot_data(df, output_dir=".", decoder_filter=None):
    if df.empty:
        print("No data to plot.")
        return

    show_objective = (decoder_filter == 'mwpf') or (decoder_filter is None)
    groups = df.groupby(['code_type', 'noise_model'])

    # Style cycles to distinguish codes with same distance
    LINE_STYLES = ['-', '--', '-.', ':']
    MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

    for (code_type, noise_model), group_df in groups:
        print(f"Plotting: {code_type} - {noise_model}")
        
        # 1. Organize Data
        variant_col = 'variant' if 'variant' in group_df.columns else None
        curve_keys = ['n', 'd', 'variant'] if variant_col else ['n', 'd']
        curves = group_df.groupby(curve_keys)
        sorted_curves = []
        for key, curve_data in curves:
            if variant_col:
                n, d, variant = key
            else:
                n, d = key
                variant = 'base'
            label, d_val = get_code_metadata(code_type, n, d)
            if code_type == 'ja25_transversal_t':
                label = f"{label} [{variant}]"
            sorted_curves.append({
                'n': n, 'd': d_val, 'variant': variant,
                'data': curve_data.sort_values('p'),
                'label': label, 'sort_key': d_val
            })

        # Sort by distance, then by n, then by variant name
        sorted_curves.sort(key=lambda x: (x['sort_key'], x['n'], x['variant']))

        # 2. Setup Standard Colors
        unique_ds = sorted(list(set(c['d'] for c in sorted_curves)))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        std_colors = prop_cycle.by_key()['color']
        d_color_map = {d: std_colors[i % len(std_colors)] for i, d in enumerate(unique_ds)}

        # Track how many times we've seen each distance to cycle styles
        d_counter = {d: 0 for d in unique_ds}

        # 3. Setup Plot
        ps = sorted(group_df['p'].unique())
        confidence_percent = int((1 - ALPHA) * 100)
        
        plt.rcParams.update({
            'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,     
            'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12
        })

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 6, height_ratios=[3, 2])
        ax_total = fig.add_subplot(gs[0, :])
        ax_obj = fig.add_subplot(gs[1, 0:3]) if show_objective else None
        ax_time = fig.add_subplot(gs[1, 3:6] if show_objective else gs[1, :])

        MAX_ERRORS = 100

        # 4. Plot Loops
        max_shots = group_df['shots'].max()
        res_limit = 1.0 / max_shots

        for i, curve in enumerate(sorted_curves):
            subset = curve['data']
            d = curve['d']

            # Color by distance
            color = d_color_map[d]

            # Marker shape cycles per variant (same-distance discriminator)
            # Linestyle is reserved for data quality only (solid/dashed/none)
            style_idx = d_counter[d]
            mk = MARKERS[style_idx % len(MARKERS)]
            d_counter[d] += 1

            p_L = subset['total_logical_error_rate'].values
            shots_val = subset['shots'].values
            errors = subset['errors'].values

            # Confidence intervals using Clopper-Pearson method
            lower_b = scipy.stats.beta.ppf(ALPHA / 2, errors, shots_val - errors + 1)
            lower_b[errors == 0] = 0.0
            upper_b = scipy.stats.beta.ppf(1 - ALPHA / 2, errors + 1, shots_val - errors)
            upper_b[errors == shots_val] = 1.0
            y_err = [p_L - lower_b, upper_b - p_L]

            # marker shape = curve identity (variant discriminator)
            # linestyle = data quality (solid >= MAX_ERRORS, dashed < MAX_ERRORS)

            # saturated (errors >= MAX_ERRORS): curve marker, solid line
            sat = errors >= MAX_ERRORS
            if sat.any():
                ax_total.errorbar(subset['p'].values[sat], p_L[sat],
                                  yerr=[y_err[0][sat], y_err[1][sat]],
                                  marker=mk, linestyle='-',
                                  label=curve['label'], color=color, capsize=3, alpha=0.8)

            # partial (0 < errors < MAX_ERRORS): curve marker, dashed line
            partial = (errors > 0) & (errors < MAX_ERRORS)
            if partial.any():
                ax_total.errorbar(subset['p'].values[partial], p_L[partial],
                                  yerr=[y_err[0][partial], y_err[1][partial]],
                                  marker=mk, linestyle='--',
                                  label=curve['label'] if not sat.any() else '_',
                                  color=color, capsize=3, alpha=0.65)

            # zero errors: downward triangle at resolution limit
            zero = errors == 0
            if zero.any():
                ax_total.scatter(subset['p'].values[zero],
                                 np.full(zero.sum(), res_limit),
                                 marker='v', color=color, alpha=0.6, s=40,
                                 label=curve['label'] if not sat.any() and not partial.any() else '_')
            
            if show_objective and 'mean_objective_per_syndrome' in subset.columns:
                ax_obj.plot(subset['p'], subset['mean_objective_per_syndrome'],
                            marker=mk, linestyle='-', color=color, label=f"d={d}")

            if 'average_cpu_time_seconds' in subset.columns:
                t_vals = subset['average_cpu_time_seconds'].dropna()
                if not t_vals.empty:
                    ax_time.plot(subset['p'][t_vals.index], t_vals.values,
                                 marker=mk, linestyle='-', color=color)

        # --- Formatting Main Plot ---
        ax_total.plot(ps, ps, linestyle='--', color='black', alpha=0.4, label="Break-even")
        
        # Unreliable Region (Dynamic)
        y_bottom = res_limit * 0.1
        
        ax_total.axhline(y=res_limit, color='gray', linestyle=':', alpha=0.5)
        x_min, x_max = min(ps), max(ps)
        ax_total.fill_between([x_min * 0.5, x_max * 1.5], y_bottom, res_limit, 
                              color='lightgray', alpha=0.3, label=f'Unreliable (< {res_limit:.1e})')

        ax_total.set_xscale('log')
        ax_total.set_yscale('log')
        ax_total.set_ylim(bottom=y_bottom)
        
        ax_total.set_xlabel('Physical Error Rate (p)')
        ax_total.set_ylabel(f'Logical Error Rate ({confidence_percent}% CI)')
        ax_total.set_title(f'Logical Error Rate: {code_type} ({noise_model})')
        ax_total.grid(True, which="both", ls="--", alpha=0.3)
        ax_total.set_xlim(min(ps)*0.8, max(ps)*1.2)

        # Data-quality legend (lower right): linestyle encodes quality, marker varies per curve
        import matplotlib.lines as mlines
        cat_handles = [
            mlines.Line2D([], [], color='gray', marker='o', linestyle='-',  label=f'≥{MAX_ERRORS} errors'),
            mlines.Line2D([], [], color='gray', marker='o', linestyle='--', label=f'<{MAX_ERRORS} errors'),
            mlines.Line2D([], [], color='gray', marker='v', linestyle='',   label='0 errors'),
        ]
        ax_total.legend(handles=cat_handles, loc='lower right',
                        title='line = data quality', fontsize=10, title_fontsize=10)
        # Curve-identity legend (upper left)
        from matplotlib.legend import Legend
        id_handles, id_labels = [], []
        for h, l in zip(*ax_total.get_legend_handles_labels()):
            if not l.startswith('_') and l not in ('Break-even', f'Unreliable (< {res_limit:.1e})'):
                id_handles.append(h)
                id_labels.append(l)
        if id_handles:
            leg2 = Legend(ax_total, id_handles, id_labels, loc='upper left',
                          title='code (color)', fontsize=10, title_fontsize=10)
            ax_total.add_artist(leg2)

        # Subplots
        if show_objective:
            ax_obj.set_xscale('log'); ax_obj.set_xlabel('p'); ax_obj.set_ylabel('Weight'); ax_obj.set_title('Mean MWPF Objective')
            ax_obj.grid(True, which="both", ls="--", alpha=0.3)
        ax_time.set_xscale('log')
        if ax_time.has_data():
            ax_time.set_yscale('log')
        ax_time.set_xlabel('p'); ax_time.set_ylabel('Seconds'); ax_time.set_title('Avg Decoding Time')
        ax_time.grid(True, which="both", ls="--", alpha=0.3)

        fig.suptitle(f'{code_type.replace("_", " ").title()} Analysis', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        out_file = os.path.join(output_dir, f"plot_{code_type}_{noise_model}.png")
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to: {out_file}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot QEC Simulation Results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .csv results")
    parser.add_argument("--output_dir", type=str, default="$HOME/Desktop/DecodingCodesWithTransversalT/fig/", help="Directory to save plots (default: project fig/)")
    parser.add_argument("--decoder", type=str, default=None, help="Filter results by decoder name (e.g. mwpf, tesseract)")
    args = parser.parse_args()

    input_path = os.path.expandvars(args.input_dir)
    output_path = os.path.expandvars(args.output_dir) if args.output_dir else input_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = load_and_aggregate_data(input_path, decoder_filter=args.decoder)
    plot_data(df, output_path, decoder_filter=args.decoder)

if __name__ == "__main__":
    main()