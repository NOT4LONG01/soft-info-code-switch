# Coverage-Restricted Random Sampling for Gap Proxy

## Summary

Added a `coverage_fraction` parameter to the 'random' gap proxy method, enabling uniform random sampling within a restricted pool of the most likely logical errors. When `coverage_fraction=0.3`, sampling is done uniformly among logical errors whose cumulative probability (sorted by likelihood) is ≤ 0.3. This feature was implemented across the core decoder, simulation utilities, simulation scripts, tests, and documentation.

## Background & Motivation

The 'random' gap proxy method samples logical classes uniformly at random to estimate the logical gap. However, uniform sampling over all 2^k logical classes becomes inefficient when most probability mass is concentrated in a small subset of likely errors.

The motivation for this feature is to focus exploration on the most likely competitors while maintaining the stochastic nature of random sampling. By restricting the sampling pool to errors within a specified cumulative probability coverage, we can:
- Reduce wasted samples on extremely unlikely logical errors
- Better estimate the gap from true competitors
- Maintain uniform sampling within the restricted pool (unlike weighted-random which samples proportionally)

## Approach & Key Ideas

**Core Algorithm:**
1. Sort logical errors by probability (descending) using the pre-computed logical error distribution
2. Compute cumulative probabilities (normalized to sum to 1.0)
3. Select only errors where cumulative probability ≤ `coverage_fraction`
4. Sample uniformly from this restricted candidate pool

**Design Decisions:**
- **Parameter semantics**: `coverage_fraction=None` or `coverage_fraction=1.0` falls back to standard uniform random sampling (no distribution required)
- **Distribution requirement**: Only require `logical_error_distribution` when `coverage_fraction < 1.0`
- **Edge case handling**: Always include at least one error (the most likely) even if its probability alone exceeds the coverage threshold
- **Integration point**: The coverage restriction is applied within `_sample_random_logical_classes()`, keeping the algorithm change localized

## Implementation Outline

**Entry Point:**
The feature is accessed through the `decode()` method of `SoftOutputsBpLsdDecoder` when `logical_gap_proxy_method='random'` and `coverage_fraction` is specified.

**Core Flow:**
1. `decode()` receives `coverage_fraction` parameter and passes it to `_compute_logical_gap_proxy()`
2. `_compute_logical_gap_proxy()` passes both `coverage_fraction` and `logical_error_distribution` to `_sample_random_logical_classes()`
3. `_sample_random_logical_classes()` validates coverage-restriction requirements (distribution provided, valid length) and delegates to `_sample_coverage_restricted_logical_classes()` when coverage restriction is active
4. The helper validates the distribution (non-zero weights), computes the restricted candidate pool, validates pool size vs `num_classes_to_explore`, and samples uniformly from it

**Simulation Integration:**
The parameter flows through `bplsd_simulation_task_single()` → `bplsd_simulation_task_parallel()` → simulation scripts (`bb_simulation.py`, `hgp_simulation.py`). The simulation scripts also update the auto-compute logic to trigger distribution pre-computation when 'random' method uses `coverage_fraction < 1.0`.

## Key Files

- `src/ldpc_post_selection/bplsd_decoder.py` - Core implementation with new `_sample_coverage_restricted_logical_classes()` helper and updated method signatures
- `simulations/utils/simulation_utils.py` - Parameter pass-through in task functions
- `simulations/bb_simulation.py` - BB code simulation with coverage_fraction config option and updated directory naming
- `simulations/hgp_simulation.py` - HGP code simulation with same updates
- `tests/test_decoder.py` - Five new tests for coverage_fraction functionality
- `docs/logical-gap-proxy-methods.md` - Documentation of the new parameter

## Tricky Parts & Gotchas

**Distribution Requirement Logic:**
The `requires_distribution` check in simulation scripts was updated to include the 'random' method when `coverage_fraction < 1.0`. Previously only 'most-likely-first' and 'weighted-random' variants required the distribution. Be careful when modifying distribution auto-compute logic—all three conditions must be considered.

**Fallback Behavior:**
`coverage_fraction=1.0` and `coverage_fraction=None` are both treated as "no restriction" and do NOT require a distribution. This is intentional to maintain backward compatibility and allow users to disable the feature without removing the parameter.

**At Least One Error:**
The implementation ensures at least one logical error is always included in the candidate pool, even if the most likely error's probability exceeds `coverage_fraction`. This prevents empty candidate pools.

**Zero-Sum Weight Validation:**
If all non-identity logical error weights sum to zero, the implementation raises a `ValueError` rather than silently falling back to uniform sampling. This strict behavior ensures users know their distribution is invalid for coverage-restricted sampling.

**Pool Size Validation:**
`num_classes_to_explore` must be >= the number of eligible errors + 1 (the initial class). If fewer classes are requested than the eligible pool size, a `ValueError` is raised. This ensures all errors within the coverage fraction are explored.

**Conditional Distribution:**
The normalization is over non-identity errors only (excluding index 0), so `coverage_fraction` represents coverage over P(error | error occurred), not the raw unconditional distribution.

**Directory Naming:**
When `coverage_fraction` is specified, simulation output directories include a `_cov{value}` suffix (e.g., `bb_minsum_iter30_lsd0_raw_gap_proxy_random_24_cov0.5`). This ensures different coverage configurations don't overwrite each other. Data collectors have been updated to recognize this naming pattern.

## Testing & Verification

Five new tests were added:
- `test_logical_gap_proxy_random_with_coverage_fraction` - Basic functionality with coverage < 1.0
- `test_logical_gap_proxy_random_coverage_fraction_one_fallback` - Verifies 1.0 falls back without needing distribution
- `test_logical_gap_proxy_random_coverage_fraction_requires_distribution` - Confirms < 1.0 raises error without distribution
- `test_logical_gap_proxy_random_coverage_fraction_invalid_values` - Tests 0.0, negative, >1.0 raise errors
- `test_logical_gap_proxy_random_coverage_fraction_small_pool` - Tests when coverage creates fewer eligible classes than requested

All 39 tests pass after implementation.

**Verification Command:**
```bash
pytest tests/test_decoder.py -v
```

## References

- **Commit SHA**: 3925408
- **Related Documentation**: `docs/logical-gap-proxy-methods.md` contains comprehensive documentation of all gap proxy methods including this new parameter
- **Plan File**: `~/.claude/plans/delegated-discovering-candy.md` contains the detailed implementation plan
