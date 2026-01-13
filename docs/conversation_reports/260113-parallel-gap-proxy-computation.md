# Parallel Execution Support for Logical Gap Proxy Computation

## Summary

This session added joblib-based parallelization to the logical gap proxy computation in the BP+LSD decoder. The `decode()` method now accepts a `num_procs_for_gap` parameter that enables parallel decoding of multiple logical classes, significantly speeding up gap proxy calculation for methods where the set of classes to explore can be determined upfront.

## Background & Motivation

Computing the logical gap proxy requires decoding multiple logical classes to find the best and second-best predictions. For methods like `random`, `most-likely-first`, and `weighted-random`, all candidate logical classes are known before decoding begins, making them naturally parallelizable. Previously, these decodings were performed sequentially, which became a bottleneck when exploring many classes.

The user requested parallelization using joblib, explicitly noting that `nearby` and adaptive methods (`most-likely-first-adaptive`, `weighted-random-adaptive`) cannot be parallelized because they determine which class to explore next based on previous decoding results.

## Approach & Key Ideas

**Parallelizable vs Non-Parallelizable Methods:**
- **Parallelizable:** `None` (exhaustive), `random`, `most-likely-first`, `weighted-random` — all classes are determined upfront
- **Non-Parallelizable:** `nearby`, `most-likely-first-adaptive`, `weighted-random-adaptive` — next class depends on previous results

**Process-Based Parallelization:**
The implementation uses `joblib.Parallel` with `prefer="processes"` rather than threads. This is necessary because the underlying `ldpc.BpLsdDecoder` maintains internal state during decoding, making thread-based parallelization unsafe.

**Fresh Decoder Instances:**
Each parallel task creates its own `SoftOutputsBpLsdDecoder` instance to avoid sharing state between workers. While this adds initialization overhead, it ensures correctness and the overhead is amortized across the parallelization speedup.

**Post-Hoc Intermediate Gap Proxies:**
When `compute_all_intermediate_gap_proxies=True` with parallel execution, intermediate gap values cannot be tracked incrementally. Instead, they are computed after all parallel decodings complete, using the order in which results were submitted (joblib preserves order by default).

## Implementation Outline

**Entry Point:**
The `decode()` method in `SoftOutputsBpLsdDecoder` accepts the new `num_procs_for_gap` parameter. When `compute_logical_gap_proxy=True`, this parameter is passed to `_compute_logical_gap_proxy()`.

**Core Components:**
- `_decode_single_logical_class()`: A module-level function (not a method) that performs decoding for a single logical class. It must be at module level to be picklable by joblib's loky backend. Creates a fresh decoder instance for thread-safety.
- `_compute_intermediate_gap_proxies_posthoc()`: Computes intermediate gap proxy values from the results list after parallel execution completes.

**Data Flow:**
1. `decode()` calls `_compute_logical_gap_proxy()` with `num_procs_for_gap`
2. For parallelizable methods, candidate logical classes are generated (sampling or enumeration)
3. If `num_procs_for_gap == 1`: sequential loop as before
4. If `num_procs_for_gap != 1`: joblib `Parallel` dispatches `_decode_single_logical_class` tasks
5. Results are collected and assembled into `explored_classes` dictionary
6. Gap proxy is computed from best and second-best LLRs

**Validation:**
At the start of `_compute_logical_gap_proxy()`, a check raises `ValueError` if `num_procs_for_gap != 1` for incompatible methods.

## Key Files

| File | Role |
|------|------|
| `src/ldpc_post_selection/bplsd_decoder.py` | Main implementation: helper functions and parallel execution branches |
| `tests/test_decoder.py` | New tests for parallel execution and error cases |
| `pyproject.toml` | Added `joblib` dependency |

## Tricky Parts & Gotchas

**Module-Level Helper Function:**
The `_decode_single_logical_class` function must be defined at module level, not as a class method. Methods cannot be pickled by joblib's default backend, causing serialization errors. This is a common gotcha with joblib parallelization.

**Intermediate Gap Proxy Ordering:**
When computing intermediate gap proxies post-hoc, the order of results matters. The implementation relies on joblib's default behavior of preserving submission order. Do not use `return_as='generator_unordered'` or similar options that would break this assumption.

**Decoder Initialization Cost:**
Each parallel task creates a new decoder instance, which involves constructing the parity check matrix and initializing the BP+LSD decoder. For very small numbers of classes to explore, the overhead may outweigh the parallelization benefit. The default `num_procs_for_gap=1` preserves sequential behavior for backwards compatibility and lets users opt-in to parallelization.

## Testing & Verification

**Automated Tests Added:**
- `test_logical_gap_proxy_parallel_random`: Verifies parallel execution produces valid results
- `test_logical_gap_proxy_parallel_most_likely_first`: Same for MLF method
- `test_logical_gap_proxy_parallel_weighted_random`: Same for weighted-random method
- `test_logical_gap_proxy_parallel_exhaustive`: Same for exhaustive (None) method
- `test_logical_gap_proxy_parallel_with_intermediate`: Verifies intermediate gap proxies work with parallel
- `test_logical_gap_proxy_parallel_raises_for_nearby`: Confirms error raised for nearby method
- `test_logical_gap_proxy_parallel_raises_for_adaptive`: Confirms error raised for adaptive methods

**Test Results:**
All 48 decoder tests pass, including 7 new parallel-specific tests.

**Manual Verification:**
```python
decoder.decode(
    syndrome,
    compute_logical_gap_proxy=True,
    logical_gap_proxy_method="random",
    num_classes_to_explore=16,
    num_procs_for_gap=4,
)
```

## Future Considerations

**Chunking for Large Numbers of Classes:**
For very large exploration budgets, it may be beneficial to implement chunked parallel execution to manage memory usage. Currently, all tasks are submitted at once.

**Adaptive Method Partial Parallelization:**
The adaptive methods could potentially be parallelized in batches (explore N classes in parallel, update best, repeat). This was not implemented as it would require careful design to maintain the adaptive behavior's benefits.

## References

- **Commit SHA:** `29d34b3`
- **Related Documentation:** `docs/logical-gap-proxy-methods.md` — comprehensive description of all gap proxy methods
