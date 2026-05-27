# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forked from `ldpc-post-selection` — a Python package for decoding quantum LDPC codes with cluster-based post-selection, implementing ["Efficient Post-Selection for General Quantum LDPC Codes"](https://arxiv.org/abs/2510.05795). The fork merges the soft-info decoder primitives and the color-code experiment pipeline into a single package organised by capability, and ships parity-check matrices + measurement schedules for 2D triangular and 3D tetrahedral color codes.

- `src/soft_info/` — single Python package, four subpackages by capability:
  - `decoders/` — soft-output decoder primitives + Sinter wrappers.
  - `codes/` — code registry, Stim circuit builders, noise models, DEM tools, leaf-factor rewrites.
  - `analysis/` — cluster labelling, sliding-window norm calculators, logical-error distribution sampling, plotting helpers.
  - `pipeline/` — CLI experiment drivers (`main`, `single_shot`, `overlapping`, `optimize_schedule`).
- `src/pcms/` — parity-check matrices (`alist` files) and optimised CNOT measurement schedules (`sched_*.json`, `*_alpha-*.json`, `memo_*.txt`) colocated per code family (`triangular_codes/`, `tetrahedral_codes/`).

Requires Python ≥ 3.11. A pre-built virtualenv lives at `env/` (Python 3.11.14).

## Common Commands

```bash
# Activate the project venv (or use env/bin/<tool> directly)
source env/bin/activate

# Install / reinstall the package (editable)
pip install -e .

# Optional: simulation suite for the original BB/HGP/surface-code analyses
pip install -e ./simulations

# Run all tests
pytest

# Run a single test
pytest tests/test_decoder.py::TestSoftOutputsBpLsdDecoder::test_decode_single_sample -v

# Run the color-code experiment pipeline (module invocation; no `cd` needed).
# Example: tetrahedral n=15 with bp_osd.
python -m soft_info.pipeline.main --code_type tetrahedral --n 15 \
    --decoder bp_osd --p_values 1e-3 5e-3 1e-2 --max_errors 1000

# Single-shot sweep (fix T, sweep W) for color codes
python -m soft_info.pipeline.single_shot tetrahedral 15 --decoder bp_osd \
    --noise phenomenological --basis X --p 0.01 --shots 1000000

# Optimise a syndrome-extraction schedule (writes into src/pcms/<code_type>_codes/)
python -m soft_info.pipeline.optimize_schedule triangular 7 --decoder bp_osd
```

`soft_info.pipeline.main` auto-detects SLURM (`SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_CPUS_PER_TASK`) and shards tasks across ranks; it writes per-rank CSVs to `data/results/<decoder>/`.

## Architecture

### `src/soft_info/` — single package, organised by capability

**Top-level public API** (`soft_info/__init__.py`): `SoftOutputsBpLsdDecoder`, `SoftOutputsMatchingDecoder`, plus the distribution helpers (`collect_logical_error_distribution`, `collect_logical_error_distribution_fast`, `logical_class_to_index`, `index_to_logical_class`, `normalize_distribution`).

#### `decoders/` — soft-output decoder primitives + Sinter wrappers

- **`base.py`** — `SoftOutputsDecoder`: base class storing parity check matrix `H`, observable matrix, and priors. Constructible from explicit matrices or from a `stim.Circuit` (via `codes/stim_tools.dem_to_parity_check`).
- **`bplsd.py`** — `SoftOutputsBpLsdDecoder`: BP+LSD decoder (`ldpc>=2.4.0`) with cluster statistics, `decode_sliding_window()` (caches window structures + per-window decoders keyed by matrix hash), and a logical-gap proxy with methods `None` / `'nearby'` / `'random'` / `'most-likely-first'` / `'weighted-random'` / `'most-likely-first-adaptive'` / `'weighted-random-adaptive'`. See `docs/logical-gap-proxy-methods.md`.
- **`matching.py`** — `SoftOutputsMatchingDecoder`: PyMatching decoder that exhaustively enumerates `2^k` logical classes to compute the true gap between best and second-best predictions.
- **`sinter.py`** — Sinter-compatible wrappers for `mwpf`, `tesseract`, `bp_osd` (via `stimbposd`), `relay_bp`, plus `build_decoder`, `read_trace`, `ALL_DECODERS`, `EXTRA_DECODERS`, `RELAY_PARAMS`. All wrappers can write a uniform 4×float32 binary trace `(cpu_time, obj_lower, obj_upper, 0.0)`. Missing optional deps degrade gracefully (`HAS_MWPF` / `HAS_TESSERACT` / `HAS_BPOSD` / `HAS_RELAYBP`).
- **`legacy.py`** — Back-compat re-exports of the historical `soft_info.decoder` names (`SoftOutputsDecoder`, `SoftOutputsBpLsdDecoder`, `SoftOutputsMatchingDecoder`, `compute_cluster_stats`); used by `simulations/` and `tests/test_decoder.py`.

#### `codes/` — code definitions, circuit builders, noise, DEM tools

- **`registry.py`** — `CodeRegistry` + `CSSCode` dataclass. **Registered families: `triangular` (2D color), `tetrahedral` (3D color)**. Other families (GO03, capped color, EQR iso/self-dual, JA25, QSD) remain defined but unregistered until alist data is added to `src/pcms/`. Distance dicts (`TETRAHEDRAL_DICT`, `TRIANGULAR_DICT`) map `n → d`. `TRIANGULAR_DIR` / `TETRAHEDRAL_DIR` resolve to `src/pcms/{triangular,tetrahedral}_codes/`. `schedule_dir(code_type)` returns the same dir — matrices and schedules live side by side.
- **`circuit.py`** — `generate_experiment_with_noise()` and `load_schedule()`. Two Stim circuit topologies chosen automatically: a self-dual schedule (shared ancilla, X-then-Z checks within a round) for `Hx == Hz` codes (triangular, GO03, QR, EQR) and a standard CSS schedule (separate X/Z ancillas) for non-self-dual codes (JA25 etc.).
- **`noise.py`** — `phenomenological_css_circuit`, `standard_depolarizing_noise_model`, `si1000_noise_model`, `bravyi_noise_model`. Phenomenological model emits detector coords `[check_idx, round_idx]` for spatial matching.
- **`stim_tools.py`** — `dem_to_parity_check()`: DEM → sparse parity-check + observable matrices. **Note (28 Jan 2026):** merges identical error mechanisms into a single column with aggregated probability (smaller Tanner graph, better BP performance — diverges from numbers in the arXiv paper). Also `remove_detectors_from_circuit()`.
- **`leaf_factored.py`** — Observable-gauge rewrite that strips leaf-qubit support from logical operators (used by `pipeline.overlapping.ler_windowed(leaf_reduce=True, …)`); collapses degeneracy between leaf-qubit data errors and boundary measurement errors.

#### `analysis/` — post-processing utilities

- **`clusters.py`** — `compute_cluster_stats`, `label_clusters` (scipy), `label_clusters_igraph` (igraph; faster), `compute_cluster_norm_fraction`, `compute_lp_norm`.
- **`sliding_window.py`** — `CommittedClusterNormCalculator`: cached sliding-window norm evaluator for multiple samples.
- **`distribution.py`** — `collect_logical_error_distribution`, `collect_logical_error_distribution_fast`, `logical_class_to_index`, `index_to_logical_class`, `normalize_distribution`. Consumed by the distribution-based gap-proxy methods in `decoders/bplsd.py`.
- **`plotting.py`** — Standalone matplotlib/seaborn helpers (legacy `tools.py`).

#### `pipeline/` — experiment drivers (CLIs)

- **`main.py`** — Sinter-driven color-code memory experiments. `python -m soft_info.pipeline.main …`. Loads codes via `CodeRegistry`, builds noisy Stim circuits, drives `sinter` sampling, writes `data/results/<decoder>/<code_tag>_<noise>_<decoder>_rank<N>.csv`. Supports resume from `data/tmp/.../resume_*.sinter`. `--decoder all` runs every available decoder.
- **`single_shot.py`** — Sweep runner for the single-shot property. `--mode w` fixes T and sweeps W (flat-tail diagnoses single-shot); `--mode t` fixes W and sweeps T (linear-growth diagnoses single-shot). Outputs `wsweep_*.csv` / `tsweep_*.csv`.
- **`overlapping.py`** — Sliding-window CSS memory decoding. `ler_windowed(code, decoder, T, W, p, n_shots, …)`. `W ≥ T` defers to a full-shot path matching `main.py`; `W < T` runs per-window DEM slicing with `syn_update` threaded across boundaries. Basis convention: `'X' → X errors on data → Hz detects → Lz`.
- **`optimize_schedule.py`** — MCTS-based (`AlphaScheduler`, arXiv:2601.12509) or ILP-based (`BaselineScheduler`) syndrome-extraction schedule optimiser. Writes to `src/pcms/<code_type>_codes/`; presence of `sched_<code_type>_val{n}[_variant].json` marks completion (re-runs are no-ops).
- The soft-info post-selection driver (per-shot gap → acceptance-rate sweep) is intended to land here in a follow-up.

#### Top-level helpers

- **`helpers.py`** — `find_logical_operator(Hx, Hz, basis)` (GF(2) null-space + coset search), `parse_and_average_stats()` (aggregates `sinter.TaskStats` + binary trace files into a DataFrame), `PROJECT_ROOT` (two dirs above `helpers.py` → repo root).

### `src/pcms/` — color-code matrices and schedules

Pure data, colocated per code family:

- `triangular_codes/`: alist matrices for n ∈ {7, 19, 37, 61, 91} with d ∈ {3, 5, 7, 9, 11} (2D color codes, self-dual → `Hx == Hz` but both files ship). Plus optimised CNOT schedules (`sched_triangular_val{n}[_variant]_<decoder>.json`, raw `triangular_n{n}[_variant]_alpha-<decoder>.json`, summary `memo_*.txt`).
- `tetrahedral_codes/`: alist matrices for n ∈ {15, 65, 175, 369} with d ∈ {3, 5, 7, 9} (3D color codes, transversal-T candidates; in general `Hx ≠ Hz`). Same schedule files.

Matrices parsed via `codes.registry.readAlist()`; schedules loaded via `codes.circuit.load_schedule()`.

### Simulations — `simulations/`

Separate `pip install -e ./simulations` package for the BB/HGP/surface-code numerical experiments from the paper. Contains `bb_simulation.py`, `bb_sliding_window_simulation.py`, `hgp_simulation.py`, `surface_code_simulation*.py`, plus `analysis/` (data collectors, notebooks, plotting helpers) and `utils/` (includes the `SlidingWindowDecoder` git submodule — clone with `--recurse-submodules`). Imports from `soft_info.decoders.legacy`, `soft_info.codes.stim_tools`, `soft_info.analysis.distribution`, etc.

### Plotting — `util/`

Standalone scripts for paper figures: `plot_comparison.py`, `plot_stim.py`, `plot_tikz.py`, `plot_tikz_single_shot.py`. Not imported by the packaged code.

## Key Dependencies

- `stim ≥ 1.14`, `ldpc ≥ 2.4.0`, `pymatching` — core decoders / DEM
- `scipy.sparse`, `numpy ≥ 2.2.4`, `python-igraph`, `numba` — performance-critical paths
- `sinter` — used by `pipeline/` for batched circuit sampling
- `galois` — GF(2) arithmetic in `soft_info/helpers.py` and `soft_info/codes/registry.py`
- Optional decoders: `mwpf`, `tesseract_decoder`, `stimbposd`, `relay_bp` (each behind a `HAS_*` flag in `soft_info/decoders/sinter.py`)

## Coding Patterns

- Parity-check matrices in the soft-info layer are `scipy.sparse.csc_matrix` with dtype `uint8`.
- `SoftOutputsBpLsdDecoder` / `SoftOutputsMatchingDecoder` accept either matrices+priors or a `stim.Circuit`; in the circuit form they auto-extract H, the observable matrix, and priors via `codes.stim_tools.dem_to_parity_check`.
- Soft outputs are returned as dicts with keys like `pred_llr`, `detector_density`, `cluster_sizes`, `cluster_llrs`, and `gap` / `gap_proxy`.
- Sliding-window decoding caches per-window decoders keyed by matrix hash — reuse the same `SoftOutputsBpLsdDecoder` instance across shots with matching window structure.
- Run pipeline scripts as `python -m soft_info.pipeline.<module>` from the repo root — relative imports resolve cleanly. Outputs land in `<repo>/data/{results,tmp}/…` (created on first run); schedules ship inside `src/pcms/` and are read directly.
