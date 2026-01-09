# Adaptive Logical Gap Proxy Methods

## Summary

This session added two new logical gap proxy computation methods: `'most-likely-first-adaptive'` and `'weighted-random-adaptive'`. These adaptive variants dynamically update the "base" logical class during exploration whenever a better class is found, enabling more efficient search of the logical class space compared to their non-adaptive counterparts.

## Background & Motivation

The logical gap proxy is a confidence metric for quantum error correction decoding that measures the difference between the best and second-best predictions across different logical classes. Computing the exact gap requires exploring all 2^k logical classes (for k observables), which is computationally expensive.

Previously, several approximation methods existed:
- **`None` (exhaustive)**: Explores all logical classes
- **`'nearby'`**: Iteratively explores adjacent classes via Hamming distance
- **`'random'`**: Uniformly samples N random classes (base-independent, simply excludes the initial class)
- **`'most-likely-first'`**: Deterministically selects the N most probable classes based on a prior distribution, using XOR with the initial best class as offset
- **`'weighted-random'`**: Samples classes with probabilities proportional to the distribution, using XOR with the initial best class as offset

The distribution-based methods (`'most-likely-first'`, `'weighted-random'`) select logical classes by XORing probable error patterns with the initial best class. In contrast, `'random'` is base-independent—it samples uniformly from all classes except the initial one. All non-adaptive methods pre-compute classes to explore upfront, then iterate through them.

The key insight motivating adaptive methods: when exploring a new logical class that turns out to be better than the current best, subsequent exploration should "recenter" around this new best class. This is because logical errors are typically correlated with the current decoding state—if we've found a better class, errors relative to **that** class are more informative than errors relative to the original class.

## Approach & Key Ideas

### Adaptive Exploration Strategy

The adaptive methods differ from their non-adaptive counterparts in one key way: they select the next class to explore based on the **current** best class, not the original best class.

The algorithm proceeds as follows:
1. Start with the initial best logical class from decoding
2. Select the next class based on the current best class as the "offset" (via XOR with logical errors)
3. Decode the new class to get its pred_llr
4. If this class is better (lower pred_llr), update the current best class
5. Repeat until `num_classes_to_explore` unique classes have been explored

Critically, the set of already-explored classes is **never reset**—only appended. When the best class changes, only the selection offset changes, not the exploration history.

### Class Selection Methods

For `'most-likely-first-adaptive'`:
- Error indices are pre-sorted by distribution (descending probability)
- Each iteration walks through sorted errors, XORing with current best class
- First resulting class not already explored is selected

For `'weighted-random-adaptive'`:
- Uses rejection sampling: sample an error from distribution, XOR with current best
- If resulting class already explored, resample (up to max retries)
- Provides stochastic exploration while biasing toward likely errors

### Comparison Table

| Aspect | Non-Adaptive | Adaptive |
|--------|--------------|----------|
| Class selection | Pre-computed from initial best | Computed iteratively from current best |
| Base class | Fixed (original) | Updates when better class found |
| Explored classes | Fresh list | Maintained set (no duplicates) |
| Selection order | Fixed upfront | Dynamic, based on current best |

## Implementation Outline

### Entry Point

The adaptive methods are invoked through the `decode()` method of `SoftOutputsBpLsdDecoder` by passing `logical_gap_proxy_method="most-likely-first-adaptive"` or `"weighted-random-adaptive"` along with required parameters `logical_error_distribution` and `num_classes_to_explore`.

### Core Components

**Helper Methods**:
- `_get_next_mlf_adaptive_class()`: For most-likely-first-adaptive, iterates through sorted error indices to find the first unexplored class
- `_sample_next_wr_adaptive_class()`: For weighted-random-adaptive, uses rejection sampling to find an unexplored class

**Main Logic in `_compute_logical_gap_proxy()`**:
- Initializes `explored_classes_set` with the original class
- Maintains `current_best_class` and `current_best_llr`
- Loops until `num_classes_to_explore` classes are in the set
- Updates current best when a better class is found
- Tracks intermediate gap proxies if requested

### Data Flow

1. `decode()` validates parameters and calls `_compute_logical_gap_proxy()`
2. Initial prediction establishes `original_logical_class` and `original_pred_llr`
3. Adaptive method block:
   - Prepares sorted indices (mlf-adaptive) or distribution probabilities (wr-adaptive)
   - Iteratively selects next class via helper method
   - Calls `_perform_fixed_logical_class_decoding()` for each class
   - Updates tracking variables
4. Common post-processing finds final best/second-best from all explored classes
5. Returns `(gap_proxy, best_pred, best_pred_llr, gap_proxies_by_num_classes)`

## Key Files

- **`src/ldpc_post_selection/bplsd_decoder.py`**: Core implementation of adaptive methods, helper functions, and validation
- **`tests/test_decoder.py`**: Five new tests covering basic functionality, intermediate gap proxies, and correct class count
- **`simulations/bb_simulation.py`**: Directory naming support for BB codes (`_mlfa_`, `_wra_` suffixes)
- **`simulations/hgp_simulation.py`**: Directory naming support for HGP codes
- **`simulations/analysis/data_collectors/collect_bb_simulation_data.py`**: Data collection support for adaptive methods
- **`simulations/analysis/data_collectors/collect_hgp_simulation_data.py`**: Data collection support for adaptive methods
- **`CLAUDE.md`**: Updated to document new methods

## Tricky Parts & Gotchas

### Explored Classes Set vs Dict

The implementation maintains both `explored_classes` (dict mapping class tuple to results) and `explored_classes_set` (set for O(1) lookup). The set is used for fast membership checking during class selection, while the dict stores the actual decoding results.

### Class Selection Never Resets

When the best class changes, **only** the offset for selecting the next class changes. The `explored_classes_set` is never cleared. This ensures we explore exactly `num_classes_to_explore` unique classes total.

### Early Termination Conditions

Both adaptive methods can terminate early:
- `most-likely-first-adaptive`: When all possible classes from the current best have been explored (helper returns `None`)
- `weighted-random-adaptive`: When rejection sampling exceeds `max_retries` (default 1000)

These conditions are rare in practice but important for correctness.

### Intermediate Gap Proxy Tracking

Intermediate gap proxies (`gap_proxy_2`, `gap_proxy_3`, etc.) are computed using a running best/second-best approach during exploration, consistent with non-adaptive methods. The final `gap_proxy` is computed from all explored classes via common post-processing.

## Testing & Verification

Five new tests were added:
1. `test_logical_gap_proxy_mlf_adaptive_method`: Basic functionality for most-likely-first-adaptive
2. `test_logical_gap_proxy_mlf_adaptive_with_intermediate`: Verifies intermediate gap proxies
3. `test_logical_gap_proxy_wr_adaptive_method`: Basic functionality for weighted-random-adaptive
4. `test_logical_gap_proxy_wr_adaptive_with_intermediate`: Verifies intermediate gap proxies
5. `test_adaptive_methods_explore_correct_number_of_classes`: Confirms both methods produce correct number of intermediate keys

All 34 decoder tests pass. To verify:
```bash
pytest tests/test_decoder.py -v
```

## Future Considerations

- The simulation infrastructure now supports adaptive methods with directory naming conventions `_mlfa_{num_classes}` and `_wra_{num_classes}`
- Performance comparison between adaptive and non-adaptive methods would be valuable for determining when adaptive exploration provides benefits
- The `max_retries` parameter for weighted-random-adaptive could be exposed as a user parameter if needed

## References

- **Commit SHA**: `d4141d1`
- **Related documentation**: `docs/conversation_logs/260108-weighted-random-gap-proxy.md` (non-adaptive weighted-random method)
- **Previous commits**:
  - `9eef325` (weighted-random method)
  - `c080827` (most-likely-first method)
