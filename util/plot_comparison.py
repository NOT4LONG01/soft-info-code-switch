"""
plot_comparison.py
------------------
Creates clustered comparison figures, one per code family:
  self_dual — go03_self_dual, eqr_self_dual_css, eqr_iso_dual, triangular, square
  css_t     — ja25_transversal_t, tetrahedral, capped_color_code

Visual encoding (Optimized for multiple dimensions):
  Color     → Distance (d)
  Marker    → Code Type and Variant (e.g., triangular, ja25 [rm])
  Linestyle → Decoder (e.g., Solid = mwpf, Dashed = tesseract)
  Alpha     → Data quality: 0.9 for ≥ MAX_ERRORS, 0.4 for < MAX_ERRORS
  Sub-panel → average decoding time

Usage:
    python util/plot_comparison.py [--results_dir ...] [--output_dir ...]
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import matplotlib.lines as mlines
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'decoder'))
from codes import GO03_DICT, EQR_ISO_DICT, EQR_SD_CSS_DICT, JA25_DICT

_DIST_DICTS = {
    'go03_self_dual':     GO03_DICT,
    'eqr_iso_dual':       EQR_ISO_DICT,
    'eqr_self_dual_css':  EQR_SD_CSS_DICT,
    'ja25_transversal_t': JA25_DICT,
}

FAMILIES = {
    'self_dual': {
        'label': 'Self-dual CSS codes',
        'codes': ['go03_self_dual', 'eqr_self_dual_css', 'eqr_iso_dual', 'triangular', 'square'],
    },
    'css_t': {
        'label': 'CSS codes with transversal T',
        'codes': ['ja25_transversal_t', 'tetrahedral', 'capped_color_code'],
    },
}

FAMILY_DECODERS  = ['mwpf', 'tesseract']
ALL_DECODERS     = ['mwpf', 'tesseract', 'relay_bp', 'bp_osd']

# Family plots: linestyle → decoder, marker → code
DECODER_LS = {
    'mwpf':      '-',
    'tesseract': '--',
    'relay_bp':  '-.',
    'bp_osd':    ':',
}

# Per-code-type plots: marker → decoder, color → (n, variant)
DECODER_MARKERS = {
    'mwpf':      'o',
    'tesseract': 's',
    'relay_bp':  '^',
    'bp_osd':    'D',
}

D_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS = ['s', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>']

MAX_ERRORS = 100
ALPHA = 0.05

GEOMETRIC_CODES = {'triangular', 'square', 'tetrahedral', 'cubic'}


def _get_d(code_type, n, d_csv):
    dist_dict = _DIST_DICTS.get(code_type)
    return dist_dict.get(n, d_csv) if dist_dict else d_csv


def load_family_data(results_root, family_codes, decoders=None):
    """
    Returns a list of dicts, one per (code_type, n, variant, decoder):
        { code_type, n, variant, decoder, df }
    df columns: p, shots, errors, p_L, d, average_cpu_time_seconds (optional)
    """
    if decoders is None:
        decoders = ALL_DECODERS
    records = []
    for code_type in family_codes:
        for decoder in decoders:
            pattern = os.path.join(
                results_root, decoder,
                f"{code_type}_n*_*_{decoder}_rank*.csv",
            )
            files = glob.glob(pattern)
            if not files:
                continue

            by_nv = defaultdict(list)
            for fpath in files:
                try:
                    df = pd.read_csv(fpath)
                    if 'n' not in df.columns or 'variant' not in df.columns:
                        continue
                    for (n, variant), grp in df.groupby(['n', 'variant']):
                        by_nv[(int(n), str(variant))].append(grp)
                except Exception:
                    pass

            for (n, variant), dfs in by_nv.items():
                raw = pd.concat(dfs, ignore_index=True)
                agg_rules = {'shots': 'sum', 'errors': 'sum'}
                if 'd' in raw.columns:
                    agg_rules['d'] = 'first'
                if 'average_cpu_time_seconds' in raw.columns:
                    agg_rules['average_cpu_time_seconds'] = 'mean'
                agg = raw.groupby('p', as_index=False).agg(agg_rules)
                if 'd' not in agg.columns:
                    agg['d'] = 0
                agg['p_L'] = agg['errors'] / agg['shots'].replace(0, np.nan)
                agg['d'] = agg['d'].apply(lambda x: _get_d(code_type, n, x))
                records.append({
                    'code_type': code_type,
                    'n': n,
                    'variant': variant,
                    'decoder': decoder,
                    'df': agg.sort_values('p'),
                })
    return records


def plot_family(records, family_label, output_path):
    if not records:
        print(f"  No data for: {family_label}")
        return

    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12,
    })

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 2])
    ax_ler  = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, :])

    # 1. Colors -> Distances
    unique_ds = sorted(set(
        r['df']['d'].iloc[0] for r in records if not r['df'].empty and 'd' in r['df'].columns
    ))
    d_color_map = {d: D_COLORS[i % len(D_COLORS)] for i, d in enumerate(unique_ds)}

    # 2. Markers -> Code Type & Variant
    unique_codes = sorted(set((r['code_type'], r['variant']) for r in records))
    marker_map = {code: MARKERS[i % len(MARKERS)] for i, code in enumerate(unique_codes)}

    all_ps = sorted(set(p for r in records for p in r['df']['p']))
    max_shots = max(r['df']['shots'].max() for r in records)
    res_limit = 1.0 / max_shots

    seen_decoders = set()

    for rec in records:
        code_type = rec['code_type']
        n         = rec['n']
        variant   = rec['variant']
        decoder   = rec['decoder']
        df        = rec['df']

        if df.empty or 'd' not in df.columns:
            continue

        d = df['d'].iloc[0]

        color = d_color_map.get(d, 'gray')
        mk    = marker_map[(code_type, variant)]
        ls    = DECODER_LS.get(decoder, '-')

        p_vals = df['p'].values
        p_L    = df['p_L'].values
        shots  = df['shots'].values
        errors = df['errors'].values

        lower = scipy.stats.beta.ppf(ALPHA / 2,     errors,          shots - errors + 1)
        upper = scipy.stats.beta.ppf(1 - ALPHA / 2, errors + 1,      shots - errors)
        lower[errors == 0] = 0.0
        upper[errors == shots] = 1.0
        y_err = [p_L - lower, upper - p_L]

        # Saturated (errors >= MAX_ERRORS): high opacity
        sat = errors >= MAX_ERRORS
        if sat.any():
            ax_ler.errorbar(
                p_vals[sat], p_L[sat],
                yerr=[y_err[0][sat], y_err[1][sat]],
                color=color, marker=mk, linestyle=ls,
                capsize=3, alpha=0.9, label='_',
            )

        # Partial (0 < errors < MAX_ERRORS): lower opacity
        partial = (errors > 0) & (errors < MAX_ERRORS)
        if partial.any():
            ax_ler.errorbar(
                p_vals[partial], p_L[partial],
                yerr=[y_err[0][partial], y_err[1][partial]],
                color=color, marker=mk, linestyle=ls,
                capsize=3, alpha=0.4, label='_',
            )

        # Zero errors: plot at resolution floor using the code's marker
        zero = errors == 0
        if zero.any():
            ax_ler.scatter(
                p_vals[zero], np.full(zero.sum(), res_limit),
                marker=mk, color=color, alpha=0.6, s=40, label='_', facecolors='none'
            )

        # Timing subplot
        if 'average_cpu_time_seconds' in df.columns:
            t_vals = df['average_cpu_time_seconds'].dropna()
            if not t_vals.empty:
                ax_time.plot(
                    df['p'].iloc[t_vals.index], t_vals.values * 1e3,
                    color=color, marker=mk, linestyle=ls, alpha=0.85, label='_',
                )

        seen_decoders.add(decoder)

    # Break-even + resolution floor
    ax_ler.plot(all_ps, all_ps, '--', color='black', alpha=0.4, label='_')
    ax_ler.axhline(res_limit, color='gray', linestyle=':', alpha=0.5)
    ax_ler.fill_between(
        [min(all_ps) * 0.8, max(all_ps) * 1.2],
        res_limit * 0.1, res_limit,
        color='lightgray', alpha=0.3,
    )

    ax_ler.set_xscale('log')
    ax_ler.set_yscale('log')
    ax_ler.set_ylim(bottom=res_limit * 0.1)
    ax_ler.set_xlim(min(all_ps) * 0.8, max(all_ps) * 1.2)
    ax_ler.set_xlabel('Physical error rate  p')
    ax_ler.set_ylabel(f'Logical error rate  ({int((1-ALPHA)*100)}% CI)')
    ax_ler.set_title(family_label)
    ax_ler.grid(True, which='both', ls='--', alpha=0.3)

    _leg_kw = dict(borderpad=1.0, labelspacing=0.6, handlelength=2.5, handletextpad=0.8)

    # Legend 1: Distance (Color) -> Top Right
    dist_handles = [
        mlines.Line2D([], [], color=d_color_map[d], marker='s', linestyle='', label=f"d={d}")
        for d in unique_ds
    ]
    leg1 = Legend(ax_ler, dist_handles, [h.get_label() for h in dist_handles],
                  loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  bbox_transform=ax_ler.transAxes,
                  title='Distance (Color)', fontsize=11, title_fontsize=11, **_leg_kw)
    ax_ler.add_artist(leg1)

    # Legend 2: Code Type (Marker) -> Middle Right (Upper)
    code_handles = []
    for (ct, var) in unique_codes:
        label = ct.replace('_', ' ') + (f" [{var}]" if var != 'base' else "")
        code_handles.append(mlines.Line2D([], [], color='gray', marker=marker_map[(ct, var)], linestyle='', label=label))
    leg2 = Legend(ax_ler, code_handles, [h.get_label() for h in code_handles],
                  loc='center left', bbox_to_anchor=(1.02, 0.65),
                  bbox_transform=ax_ler.transAxes,
                  title='Code Type (Marker)', fontsize=10, title_fontsize=11, **_leg_kw)
    ax_ler.add_artist(leg2)

    # Legend 3: Decoder (Linestyle) -> Middle Right (Lower)
    dec_handles = [
        mlines.Line2D([], [], color='black', marker='', linestyle=DECODER_LS[dec], label=dec)
        for dec in FAMILY_DECODERS if dec in seen_decoders
    ]
    leg3 = Legend(ax_ler, dec_handles, [h.get_label() for h in dec_handles],
                  loc='center left', bbox_to_anchor=(1.02, 0.3),
                  bbox_transform=ax_ler.transAxes,
                  title='Decoder (Line)', fontsize=11, title_fontsize=11, **_leg_kw)
    ax_ler.add_artist(leg3)

    # Legend 4: Data Quality (Alpha) -> Bottom Right
    quality_handles = [
        mlines.Line2D([], [], color='gray', marker='', linestyle='-', linewidth=3, alpha=0.9, label=f'≥{MAX_ERRORS} errors'),
        mlines.Line2D([], [], color='gray', marker='', linestyle='-', linewidth=3, alpha=0.4, label=f'<{MAX_ERRORS} errors'),
    ]
    ax_ler.legend(handles=quality_handles, loc='lower left',
                  bbox_to_anchor=(1.02, 0.0), bbox_transform=ax_ler.transAxes,
                  title='Data Quality (Opacity)', fontsize=10, title_fontsize=11, **_leg_kw)

    # Timing subplot
    ax_time.set_xscale('log')
    if ax_time.has_data():
        ax_time.set_yscale('log')
    ax_time.set_xlabel('Physical error rate  p')
    ax_time.set_ylabel('Avg decode time (ms / shot)')
    ax_time.set_title('Decoder speed')
    ax_time.grid(True, which='both', ls='--', alpha=0.3)

    fig.suptitle(f'Decoder & Code Comparison — {family_label}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def _code_label(code_type, n, variant, d):
    """Label for a single (n, variant) curve in a per-code-type plot."""
    if code_type in GEOMETRIC_CODES:
        label = f"d={d}"
    else:
        label = f"n={n}, d={d}"
    if variant and variant != 'base':
        label += f"  [{variant}]"
    return label


def plot_codetype(records, title, output_path):
    """Per-code-type plot: color → (n, variant), marker → decoder."""
    if not records:
        print(f"  No data for: {title}")
        return

    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12,
    })

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 2])
    ax_ler  = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, :])

    # Assign colors to each unique (n, variant)
    curve_ids = sorted(set((r['n'], r['variant']) for r in records),
                       key=lambda x: (x[0], x[1]))
    color_map = {cid: D_COLORS[i % len(D_COLORS)] for i, cid in enumerate(curve_ids)}

    all_ps = sorted(set(p for r in records for p in r['df']['p']))
    max_shots = max(r['df']['shots'].max() for r in records)
    res_limit = 1.0 / max_shots

    seen_decoders = set()
    code_type = records[0]['code_type']

    for rec in records:
        n       = rec['n']
        variant = rec['variant']
        decoder = rec['decoder']
        df      = rec['df']

        color = color_map[(n, variant)]
        mk    = DECODER_MARKERS.get(decoder, 'o')

        p_vals = df['p'].values
        p_L    = df['p_L'].values
        shots  = df['shots'].values
        errors = df['errors'].values

        lower = scipy.stats.beta.ppf(ALPHA / 2,     errors,          shots - errors + 1)
        upper = scipy.stats.beta.ppf(1 - ALPHA / 2, errors + 1,      shots - errors)
        lower[errors == 0] = 0.0
        upper[errors == shots] = 1.0
        y_err = [p_L - lower, upper - p_L]

        sat = errors >= MAX_ERRORS
        if sat.any():
            ax_ler.errorbar(
                p_vals[sat], p_L[sat],
                yerr=[y_err[0][sat], y_err[1][sat]],
                color=color, marker=mk, linestyle='-',
                capsize=3, alpha=0.9, label='_',
            )

        partial = (errors > 0) & (errors < MAX_ERRORS)
        if partial.any():
            ax_ler.errorbar(
                p_vals[partial], p_L[partial],
                yerr=[y_err[0][partial], y_err[1][partial]],
                color=color, marker=mk, linestyle='--',
                capsize=3, alpha=0.5, label='_',
            )

        zero = errors == 0
        if zero.any():
            ax_ler.scatter(
                p_vals[zero], np.full(zero.sum(), res_limit),
                marker='v', color=color, alpha=0.6, s=40, label='_',
            )

        if 'average_cpu_time_seconds' in df.columns:
            t_vals = df['average_cpu_time_seconds'].dropna()
            if not t_vals.empty:
                ax_time.plot(
                    df['p'].iloc[t_vals.index], t_vals.values * 1e3,
                    color=color, marker=mk, linestyle='-', alpha=0.85, label='_',
                )

        seen_decoders.add(decoder)

    ax_ler.plot(all_ps, all_ps, '--', color='black', alpha=0.4, label='_')
    ax_ler.axhline(res_limit, color='gray', linestyle=':', alpha=0.5)
    ax_ler.fill_between(
        [min(all_ps) * 0.8, max(all_ps) * 1.2],
        res_limit * 0.1, res_limit, color='lightgray', alpha=0.3,
    )

    ax_ler.set_xscale('log')
    ax_ler.set_yscale('log')
    ax_ler.set_ylim(bottom=res_limit * 0.1)
    ax_ler.set_xlim(min(all_ps) * 0.8, max(all_ps) * 1.2)
    ax_ler.set_xlabel('Physical error rate  p')
    ax_ler.set_ylabel(f'Logical error rate  ({int((1-ALPHA)*100)}% CI)')
    ax_ler.set_title(title)
    ax_ler.grid(True, which='both', ls='--', alpha=0.3)

    _leg_kw = dict(borderpad=1.0, labelspacing=0.6, handlelength=2.5, handletextpad=0.8,
                   bbox_transform=ax_ler.transAxes)

    # Legend 1 (top right): code size — color
    size_handles = []
    for (n, variant) in curve_ids:
        color = color_map[(n, variant)]
        d_rep = next(r['df']['d'].iloc[0] for r in records
                     if r['n'] == n and r['variant'] == variant)
        label = _code_label(code_type, n, variant, d_rep)
        size_handles.append(
            mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=label)
        )
    leg1 = Legend(ax_ler, size_handles, [h.get_label() for h in size_handles],
                  loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  title='Code size  (color)', fontsize=11, title_fontsize=11, **_leg_kw)
    ax_ler.add_artist(leg1)

    # Legend 2 (middle right): decoder — marker
    dec_handles = [
        mlines.Line2D([], [], color='black', marker=DECODER_MARKERS[d], linestyle='-', label=d)
        for d in ALL_DECODERS if d in seen_decoders
    ]
    leg2 = Legend(ax_ler, dec_handles, [h.get_label() for h in dec_handles],
                  loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title='Decoder  (marker)', fontsize=11, title_fontsize=11, **_leg_kw)
    ax_ler.add_artist(leg2)

    # Legend 3 (bottom right): data quality — linestyle
    quality_handles = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='-',  label=f'≥{MAX_ERRORS} errors'),
        mlines.Line2D([], [], color='gray', marker='o', linestyle='--', label=f'<{MAX_ERRORS} errors'),
        mlines.Line2D([], [], color='gray', marker='v', linestyle='',   label='0 errors (upper limit)'),
    ]
    ax_ler.legend(handles=quality_handles, loc='lower left',
                  bbox_to_anchor=(1.02, 0.0),
                  title='Data quality  (linestyle)', fontsize=11, title_fontsize=11, **_leg_kw)

    ax_time.set_xscale('log')
    if ax_time.has_data():
        ax_time.set_yscale('log')
    ax_time.set_xlabel('Physical error rate  p')
    ax_time.set_ylabel('Avg decode time (ms / shot)')
    ax_time.set_title('Decoder speed')
    ax_time.grid(True, which='both', ls='--', alpha=0.3)

    fig.suptitle(f'Decoder comparison — {title}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--output_dir',  default=None)
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), '..')
    results_root = args.results_dir or os.path.join(project_root, 'data', 'results')
    output_dir   = args.output_dir  or os.path.join(project_root, 'fig', 'comparison')

    # One plot per family — restricted to mwpf and tesseract
    for family_key, family_info in FAMILIES.items():
        records = load_family_data(results_root, family_info['codes'],
                                   decoders=FAMILY_DECODERS)
        out = os.path.join(output_dir, f"plot_comparison_{family_key}_depolarizing.png")
        plot_family(records, family_info['label'], out)

    # One plot per code_type — all decoders, marker = decoder
    all_code_types = [ct for fam in FAMILIES.values() for ct in fam['codes']]
    for code_type in all_code_types:
        records = load_family_data(results_root, [code_type],
                                   decoders=ALL_DECODERS)
        if not records:
            continue
        title = code_type.replace('_', ' ').title()
        out = os.path.join(output_dir, f"plot_comparison_{code_type}_depolarizing.png")
        plot_codetype(records, title, out)


if __name__ == '__main__':
    main()