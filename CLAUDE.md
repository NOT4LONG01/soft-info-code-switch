# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

Forked from `ldpc-post-selection` ([paper](https://arxiv.org/abs/2510.05795)). The fork merges the soft-info decoder primitives with a Sinter-driven color-code experiment pipeline into a single package `soft_info`, organised by capability, and ships matrices + measurement schedules for 2D triangular and 3D tetrahedral color codes.

Requires Python ≥ 3.11. A working venv is at `env/` (Python 3.11.14, `pip install -e .` already applied).

## Layout

```
src/
  soft_info/                  # the package
    __init__.py               # re-exports top-level API
    helpers.py                # PROJECT_ROOT, find_logical_operator, parse_and_average_stats
    decoders/                 # soft-output decoder primitives + Sinter wrappers
      base.py, bplsd.py, matching.py, sinter.py, legacy.py
    codes/                    # code registry, circuit builders, noise, DEM tools
      registry.py, circuit.py, noise.py, stim_tools.py, leaf_factored.py
    analysis/                 # post-processing utilities
      clusters.py, sliding_window.py, distribution.py, plotting.py
    pipeline/                 # CLI experiment drivers
      main.py, single_shot.py, overlapping.py, optimize_schedule.py
  pcms/                       # alist matrices + measurement schedules (per family)
    triangular_codes/         # n ∈ {7,19,37,61,91}, d ∈ {3,5,7,9,11}
    tetrahedral_codes/        # n ∈ {15,65,175,369}, d ∈ {3,5,7,9}
tests/                        # test_decoder.py, test_logical_error_distribution.py
simulations/                  # separate pip-installable package for BB/HGP/surface-code analyses
util/                         # standalone paper-figure scripts (not imported)
examples/basic_usage.ipynb    # entry-point notebook
docs/                         # logical-gap-proxy-methods.md + conversation reports
env/                          # local venv (gitignored)
data/                         # pipeline outputs (gitignored)
```

`simulations/utils/SlidingWindowDecoder` is a git submodule — clone with `--recurse-submodules`.

## Common commands

```bash
# Activate the venv (or call env/bin/<tool> directly)
source env/bin/activate

# Install / reinstall the package
pip install -e .
pip install -e ./simulations          # optional, for the paper's BB/HGP/surface-code runs

# Tests
pytest                                                                    # all
pytest tests/test_decoder.py::TestSoftOutputsBpLsdDecoder -v             # one class

# Color-code memory experiment (Sinter-driven, SLURM-aware)
python -m soft_info.pipeline.main --code_type tetrahedral --n 15 \
    --decoder bp_osd --p_values 1e-3 5e-3 1e-2 --max_errors 1000

# Single-shot sweep (--mode w: fix T, sweep W; --mode t: fix W, sweep T)
python -m soft_info.pipeline.single_shot tetrahedral 15 --decoder bp_osd \
    --noise phenomenological --basis X --p 0.01 --shots 1000000

# Optimise a syndrome-extraction schedule (writes into src/pcms/<code_type>_codes/)
python -m soft_info.pipeline.optimize_schedule triangular 7 --decoder bp_osd
```

`soft_info.pipeline.main` reads `SLURM_PROCID` / `SLURM_NTASKS` to shard tasks across ranks; per-rank CSVs land in `data/results/<decoder>/`, with resume state in `data/tmp/`.

## What each subpackage holds

### `soft_info.decoders`
- **`base.SoftOutputsDecoder`** — stores `H`, `obs_matrix`, `priors`; constructible from matrices or from a `stim.Circuit` (auto-extracts via `codes.stim_tools.dem_to_parity_check`).
- **`bplsd.SoftOutputsBpLsdDecoder`** — BP+LSD (`ldpc>=2.4.0`) with cluster stats, `decode_sliding_window()` (caches per-window decoders by matrix hash), and gap-proxy methods `None`/`nearby`/`random`/`most-likely-first`/`weighted-random` (+ `-adaptive` variants). See [docs/logical-gap-proxy-methods.md](docs/logical-gap-proxy-methods.md).
- **`matching.SoftOutputsMatchingDecoder`** — PyMatching enumerating `2^k` logical classes for the true gap.
- **`sinter`** — wrappers for `mwpf`, `tesseract`, `bp_osd` (via `stimbposd`), `relay_bp`. `build_decoder(name, trace_filename=…)` returns a `sinter.Decoder`. Each wrapper optionally appends 16-byte records `(cpu_time, obj_lower, obj_upper, 0.0)` to `trace_filename`; `read_trace(path)` loads them as a DataFrame. Missing optional deps degrade gracefully (`HAS_MWPF` / `HAS_TESSERACT` / `HAS_BPOSD` / `HAS_RELAYBP` / `HAS_PYMATCHING` / `HAS_FUSION_BLOSSOM`). `ALL_DECODERS = ["mwpf", "tesseract"]`; `EXTRA_DECODERS = ["bp_osd", "relay_bp"]`.
- **`legacy`** — back-compat re-exports (`SoftOutputsDecoder`, `SoftOutputsBpLsdDecoder`, `SoftOutputsMatchingDecoder`, `compute_cluster_stats`) for callers of the historical `soft_info.decoder` namespace (used by `simulations/` and `tests/test_decoder.py`).

### `soft_info.codes`
- **`registry`** — `CodeRegistry`, `CSSCode` dataclass, `schedule_dir(code_type)`, `readAlist()`, distance dicts (`TRIANGULAR_DICT`, `TETRAHEDRAL_DICT`), data dirs (`TRIANGULAR_DIR`, `TETRAHEDRAL_DIR`). **Registered families: `triangular`, `tetrahedral`.** Other classes (GO03, capped color, EQR iso/self-dual, JA25, QSD) remain defined but unregistered until their alist data is dropped into `src/pcms/`.
- **`circuit`** — `generate_experiment_with_noise()` + `load_schedule()`. Picks a self-dual schedule (shared ancilla, X-then-Z within a round) when `Hx == Hz`; otherwise standard CSS (separate X/Z ancillas).
- **`noise`** — `phenomenological_css_circuit`, `standard_depolarizing_noise_model`, `si1000_noise_model`, `bravyi_noise_model`. Phenomenological model writes detector coords `[check_idx, round_idx]`.
- **`stim_tools`** — `dem_to_parity_check(dem, merge_duplicates=True)` → `(H, obs_matrix, p)`. *Note (28 Jan 2026):* duplicate error mechanisms are merged into a single column with XOR-combined probability (smaller Tanner graph, better BP — diverges from numbers in the arXiv paper). Also `remove_detectors_from_circuit()`.
- **`leaf_factored`** — `reduce_Lz_leaf_support`, `leaf_mask` — observable-gauge rewrite collapsing leaf-qubit / boundary-measurement degeneracy (used by `ler_windowed(leaf_reduce=True, …)`).

### `soft_info.analysis`
- **`clusters`** — `compute_cluster_stats`, `label_clusters` (scipy), `label_clusters_igraph` (faster), `compute_cluster_norm_fraction`, `compute_lp_norm`.
- **`sliding_window`** — `CommittedClusterNormCalculator` (cached evaluator for many samples).
- **`distribution`** — `collect_logical_error_distribution(_fast)`, `logical_class_to_index`, `index_to_logical_class`, `normalize_distribution`. Consumed by BP+LSD's distribution-based gap proxies.
- **`plotting`** — matplotlib/seaborn helpers (legacy `tools.py`).

### `soft_info.pipeline`
- **`main`** — Sinter color-code memory runs. Writes `data/results/<decoder>/<code_tag>_<noise>_<decoder>_rank<N>.csv`, resumes from `data/tmp/.../resume_*.sinter`. `--decoder all` runs every available decoder.
- **`single_shot`** — single-shot-property sweep; outputs `wsweep_*.csv` / `tsweep_*.csv`. Flat tail (mode w) or linear growth (mode t) diagnose the single-shot threshold.
- **`overlapping`** — `ler_windowed(code, decoder, T, W, p, n_shots, …)`. `W ≥ T` defers to a full-shot path matching `main`; `W < T` slices the DEM per window with `syn_update` threaded across boundaries. Basis convention: `'X' → X errors on data → Hz detects → Lz`. Imports `quits` (external dep, only needed for this driver).
- **`optimize_schedule`** — MCTS (`AlphaScheduler`, arXiv:2601.12509) or ILP (`BaselineScheduler`) syndrome-extraction optimiser. Writes to `src/pcms/<code_type>_codes/`; presence of `sched_<code_type>_val{n}[_variant].json` is the completion marker. Imports `asyndrome` (external dep, only needed for this driver).
- *Planned:* per-shot soft-info post-selection driver (gap → acceptance-rate sweep).

### `src/pcms/<family>_codes/`
- `n{N}_d{D}_H{x,z}.alist` — parity-check matrices.
- `sched_<family>_val{n}[_variant]_<decoder>.json` — compact schedule loaded by `circuit.load_schedule()`.
- `<family>_n{n}[_variant]_alpha-<decoder>.json` — raw AlphaSyndrome schedule.
- `memo_<family>_n{n}[_variant]_alpha-<decoder>.txt` — human-readable summary written by the optimiser.

## Dependencies

Required (in `pyproject.toml`): `stim≥1.14`, `ldpc≥2.4.0`, `pymatching`, `scipy`, `numpy≥2.2.4`, `python-igraph`, `numba`, `galois`, `sinter`, `joblib`, `matplotlib`, `seaborn`, `tqdm`, `cython`, `pytest`.

Optional decoder backends (each behind a `HAS_*` flag in `decoders/sinter.py`): `mwpf`, `tesseract_decoder`, `stimbposd`, `relay_bp`, plus `pymatching`, `fusion_blossom`.

External-only deps needed by specific pipeline modules: `quits` (for `pipeline.overlapping`), `asyndrome` (for `pipeline.optimize_schedule`). The `simulations` subpackage installs `quits` via its own `pyproject.toml`.

## Coding patterns

- Parity-check matrices in the soft-info layer are `scipy.sparse.csc_matrix` dtype `uint8`.
- `SoftOutputs*Decoder` accept either matrices+priors or a `stim.Circuit`; in circuit form they auto-extract H, obs matrix, and priors via `codes.stim_tools.dem_to_parity_check`.
- Soft outputs come back as dicts keyed by `pred_llr`, `detector_density`, `cluster_sizes`, `cluster_llrs`, `gap` / `gap_proxy`.
- For sliding-window decoding, reuse the same `SoftOutputsBpLsdDecoder` instance across shots — per-window decoders are cached by matrix hash.
- Run pipeline scripts as `python -m soft_info.pipeline.<module>` from the repo root; relative imports resolve cleanly and outputs go to `<repo>/data/`.
