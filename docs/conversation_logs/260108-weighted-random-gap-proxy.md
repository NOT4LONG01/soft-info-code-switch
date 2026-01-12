# Weighted-Random Logical Gap Proxy Method

## Summary

This session added a new `'weighted-random'` logical gap proxy computation method to the BP+LSD decoder. Unlike uniform random sampling (`'random'`) or deterministic selection (`'most-likely-first'`), this method samples logical classes with probabilities proportional to a provided logical error distribution. Additionally, the implementation was refactored to consolidate the best-selection logic into a single common post-processing block, ensuring all methods consistently return the correction with the minimum weight (lowest `pred_llr`).

## Background & Motivation

The logical gap proxy is a confidence metric for quantum error correction decoding. It measures the difference between the best and second-best decoding predictions across different logical classes. Computing the exact gap requires exploring all 2^k logical classes (for k observables), which is computationally expensive.

Previously, three approximation methods existed:
- **`None` (exhaustive)**: Explores all logical classes
- **`'nearby'`**: Iteratively explores adjacent classes
- **`'random'`**: Uniformly samples N random classes
- **`'most-likely-first'`**: Deterministically selects the N most probable classes based on a prior distribution

The `'weighted-random'` method fills a gap between `'random'` and `'most-likely-first'`:
- Like `'random'`, it's stochastic (non-deterministic)
- Like `'most-likely-first'`, it uses the logical error distribution to guide selection
- It provides a middle ground: biased toward likely errors but with randomness for exploration

| Method | Selection Strategy | Deterministic? | Uses Distribution? |
|--------|-------------------|----------------|-------------------|
| `'random'` | Uniform sampling | No | No |
| `'most-likely-first'` | Top N by probability | Yes | Yes |
| `'weighted-random'` | Probability-weighted sampling | No | Yes |

## Approach & Key Ideas

### Weighted Sampling

The method samples logical error indices without replacement, using probabilities proportional to the distribution values. The identity error (index 0) is excluded since applying it doesn't change the logical class. Sampled error patterns are XORed with the initial best logical class to obtain alternative classes to explore.

### Distribution Auto-Normalization

Users can provide either raw counts or pre-normalized probabilities. The implementation normalizes internally by dividing by the sum of weights. If all weights are zero, it falls back to uniform sampling with a warning.

### Common Post-Processing Refactoring

Before this change, `'random'`, `'most-likely-first'`, and `'weighted-random'` each had inline tracking of best/second-best with early returns, while `'nearby'` and `None` used common post-processing code. This duplication risked inconsistent behavior.

The refactoring ensures all methods:
1. Populate `explored_classes` with their exploration results
2. Fall through to a single common post-processing block
3. Return the correction with minimum `pred_llr` (not necessarily the initial prediction)

This guarantees that if an alternative logical class yields a lower-weight correction than the initial best, that correction is returned.

## Implementation Outline

### Entry Point

The `decode()` method in `SoftOutputsBpLsdDecoder` accepts `logical_gap_proxy_method="weighted-random"` along with `logical_error_distribution` and `num_classes_to_explore` parameters.

### Core Components

**`_sample_weighted_random_logical_classes()`**: New method that:
- Takes the best logical class and distribution as input
- Normalizes distribution weights (excluding identity)
- Uses `np.random.choice()` with `replace=False` and probability weights
- Converts sampled error indices to logical classes via XOR

**`_compute_logical_gap_proxy()`**: Modified to:
- Initialize `gap_proxies_by_num_classes` before method branches
- Remove early returns from sampling-based methods
- Track intermediate gap proxies during iteration (for `compute_all_intermediate_gap_proxies`)
- Use common post-processing for final best/second-best selection

### Data Flow

1. Initial decoding produces `original_pred_llr` and `pred`
2. Method-specific code samples/selects logical classes to explore
3. Each alternative class is decoded via `_perform_fixed_logical_class_decoding()`
4. Results stored in `explored_classes` dict
5. Common code finds minimum `pred_llr`, extracts corresponding `best_pred`
6. Returns `(gap_proxy, best_pred, best_pred_llr, gap_proxies_by_num_classes)`

## Key Files

- **`src/ldpc_post_selection/bplsd_decoder.py`**: Core implementation of `_sample_weighted_random_logical_classes()` and refactored `_compute_logical_gap_proxy()`
- **`tests/test_decoder.py`**: Unit tests for the new method (5 new test functions)
- **`simulations/bb_simulation.py`**: Directory naming for 'weighted-random' simulations (`_wr_` suffix)
- **`simulations/hgp_simulation.py`**: Same directory naming support for HGP codes
- **`CLAUDE.md`**: Updated to document the new method

## Tricky Parts & Gotchas

### Intermediate Gap Proxy Tracking

The intermediate gap proxies (`gap_proxy_2`, `gap_proxy_3`, etc.) are computed during iteration based on the running best/second-best seen so far. This tracking must happen inside the exploration loop, even though the final result comes from common post-processing. The tracking code only runs when `compute_all_intermediate_gap_proxies=True`.

### All-Zero Distribution Edge Case

If the provided distribution has all zeros (or only the identity has non-zero weight), the method falls back to uniform sampling rather than raising an error. A warning is printed in verbose mode.

### Validation Sharing

Both `'most-likely-first'` and `'weighted-random'` require the same parameters (`logical_error_distribution` and `num_classes_to_explore`). The validation logic is shared using:
```
if logical_gap_proxy_method in ("most-likely-first", "weighted-random"):
```

### Return Value Consistency

The common post-processing ensures that if any explored class has a lower `pred_llr` than the initial prediction, that class's correction becomes `best_pred`. This is important: the method should return the globally best correction found, not just the initial one.

## Testing & Verification

Five new test functions were added:

1. **`test_logical_gap_proxy_weighted_random_method`**: Basic functionality test
2. **`test_logical_gap_proxy_weighted_random_with_intermediate`**: Verifies intermediate gap proxies
3. **`test_logical_gap_proxy_weighted_random_invalid_params`**: Validates error handling for missing/invalid parameters
4. **`test_sample_weighted_random_logical_classes`**: Tests the sampling helper method directly
5. **`test_weighted_random_all_zero_weights_fallback`**: Confirms graceful fallback to uniform sampling

All 29 decoder tests pass. The full test suite shows 66 passed (2 pre-existing failures unrelated to this work).

## Future Considerations

- The simulation infrastructure now supports `'weighted-random'` with directory naming convention `_wr_{num_classes}`, ready for numerical experiments
- Data collection scripts may need updating to support this method (similar to the existing `'most-likely-first'` support)

## References

- **Commit SHA**: `9eef325`
- **Related commits**:
  - `1e84804` (most-likely-first method)
  - `8da277f` (method selector consolidation)
- **Documentation**: `docs/conversation_logs/most-likely-first-method.md` (related method documentation)
