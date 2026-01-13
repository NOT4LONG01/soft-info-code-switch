# Most-Likely-First Logical Gap Proxy Method

## Overview

The `'most-likely-first'` method is a new logical gap proxy computation strategy for quantum LDPC decoders. It leverages empirically-obtained logical error distributions to intelligently select which logical classes to explore, prioritizing those most likely to compete with the decoder's best prediction.

This document explains the motivation, theory, implementation, and usage of this method.

## Table of Contents

1. [Background: Logical Gap Proxy](#background-logical-gap-proxy)
2. [Motivation](#motivation)
3. [Key Intuition](#key-intuition)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Implementation Details](#implementation-details)
6. [Usage Guide](#usage-guide)
7. [Collecting Logical Error Distributions](#collecting-logical-error-distributions)
8. [Testing](#testing)
9. [Performance Considerations](#performance-considerations)
10. [Comparison with Other Methods](#comparison-with-other-methods)

---

## Background: Logical Gap Proxy

In quantum error correction, the **logical gap** measures the confidence of a decoder's prediction. For a decoder predicting logical class $c^*$, the logical gap is:

$$\text{gap} = \min_{c \neq c^*} \text{LLR}(c) - \text{LLR}(c^*)$$

where $\text{LLR}(c)$ is the log-likelihood ratio of the decoder's best error pattern consistent with logical class $c$.

Computing the exact logical gap requires exploring all $2^k$ logical classes (where $k$ is the number of logical qubits), which becomes computationally prohibitive for codes with many logical qubits.

**Logical gap proxy** methods approximate the gap by exploring only a subset of logical classes. Existing methods include:

- **`None` (exhaustive)**: Explores all $2^k$ classes - exact but expensive
- **`'nearby'`**: Explores adjacent classes (single-bit flips) using BFS
- **`'random'`**: Randomly samples logical classes

---

## Motivation

The `'random'` method samples logical classes uniformly, treating all non-predicted classes as equally likely competitors. However, in practice:

1. **Logical errors are not uniform**: Some logical error patterns occur much more frequently than others
2. **The gap is determined by the closest competitor**: If we know which errors are most likely, we should prioritize exploring those classes first
3. **Limited exploration budget**: When we can only explore $n \ll 2^k$ classes, the choice of which classes to explore significantly impacts the quality of the gap proxy estimate

The `'most-likely-first'` method addresses these issues by using empirical knowledge about logical error frequencies to guide the exploration.

---

## Key Intuition

The core insight is that the **logical gap proxy should prioritize exploring classes that are most likely to be the "second best" prediction**.

Consider a decoder that predicts logical class $c^*$. The true gap is determined by the logical class $c'$ that has the smallest LLR among all $c \neq c^*$. If we know from historical data that certain logical errors (transitions from the true class to a decoded class) occur more frequently, we can infer which competing classes are most likely to have low LLRs.

**The algorithm:**

1. **Obtain the best logical class** $c^*$ from the decoder's initial prediction
2. **Rank logical errors** by their empirical frequency/probability (excluding the identity error)
3. **Generate candidate classes** by applying the most likely errors to $c^*$:
   - If error $e$ is likely, then class $c^* \oplus e$ (XOR) is likely to be a strong competitor
4. **Explore these classes first** to quickly find the true second-best prediction

This is analogous to how a chess engine might prioritize analyzing the most threatening opponent moves rather than sampling moves uniformly.

---

## Mathematical Formulation

### Notation

- $k$: Number of logical qubits (observables)
- $c \in \{0, 1\}^k$: Logical class (k-bit vector)
- $c^*$: Best predicted logical class from initial decoding
- $e \in \{0, 1\}^k$: Logical error pattern
- $P(e)$: Probability/frequency of logical error $e$ (from empirical distribution)
- $\text{LLR}(c)$: Log-likelihood ratio for the best error pattern in class $c$

### Index Encoding

Logical classes and errors are encoded as integers using the convention:

$$\text{index}(c) = \sum_{j=0}^{k-1} c_j \cdot 2^j$$

where $c_j$ is the $j$-th bit. Index 0 corresponds to the all-zeros pattern (identity/no error).

### Algorithm

Given:
- Best logical class $c^*$
- Logical error distribution $\{P(e)\}_{e \in \{0,1\}^k}$
- Number of classes to explore $n$

**Step 1**: Sort error indices by probability (descending):
$$e_1, e_2, \ldots, e_{2^k-1} \quad \text{where} \quad P(e_1) \geq P(e_2) \geq \cdots$$

**Step 2**: Exclude the identity error (index 0) since $c^* \oplus 0 = c^*$

**Step 3**: Select top $n-1$ errors: $e_1, e_2, \ldots, e_{n-1}$

**Step 4**: Generate classes to explore:
$$\mathcal{C} = \{c^* \oplus e_i : i = 1, \ldots, n-1\}$$

**Step 5**: For each $c \in \mathcal{C}$, perform fixed-class decoding to obtain $\text{LLR}(c)$

**Step 6**: Compute gap proxy:
$$\text{gap\_proxy} = \min_{c \in \mathcal{C} \cup \{c^*\}, c \neq c^*_{\text{best}}} \text{LLR}(c) - \text{LLR}(c^*_{\text{best}})$$

where $c^*_{\text{best}}$ is the class with minimum LLR among all explored classes.

---

## Implementation Details

### File Locations

- **Core implementation**: `src/ldpc_post_selection/bplsd_decoder.py`
- **Distribution collection**: `src/ldpc_post_selection/logical_error_distribution.py`
- **Tests**: `tests/test_decoder.py`, `tests/test_logical_error_distribution.py`

### Key Components

#### 1. Index-to-Class Conversion (`_index_to_logical_class`)

```python
def _index_to_logical_class(self, index: int, num_observables: int) -> np.ndarray:
    """Convert integer index to logical class bit pattern."""
    if num_observables <= 64:
        bit_positions = np.arange(num_observables, dtype=np.uint64)
        logical_class = ((np.uint64(index) >> bit_positions) & 1).astype(bool)
    else:
        # Fallback for >64 observables
        logical_class = np.array(
            [(index >> j) & 1 for j in range(num_observables)], dtype=bool
        )
    return logical_class
```

#### 2. Most-Likely Class Selection (`_get_most_likely_logical_classes`)

```python
def _get_most_likely_logical_classes(
    self,
    best_logical_class: np.ndarray,
    logical_error_distribution: np.ndarray,
    num_classes_to_explore: int,
    verbose: bool = False,
) -> List[np.ndarray]:
    """Get logical classes based on most likely logical errors."""
    # Sort by probability (descending), exclude identity (index 0)
    error_indices_sorted = np.argsort(logical_error_distribution)[::-1]
    non_identity_sorted = error_indices_sorted[error_indices_sorted != 0]
    selected_error_indices = non_identity_sorted[:num_classes_to_explore - 1]

    # Apply errors to best class via XOR
    logical_classes = []
    for error_idx in selected_error_indices:
        error_pattern = self._index_to_logical_class(int(error_idx), num_observables)
        resulting_class = best_logical_class ^ error_pattern
        logical_classes.append(resulting_class)

    return logical_classes
```

#### 3. Gap Proxy Computation Branch

The `'most-likely-first'` branch in `_compute_logical_gap_proxy()`:

1. Calls `_get_most_likely_logical_classes()` to get candidate classes
2. Iteratively decodes each class using `_perform_fixed_logical_class_decoding()`
3. Tracks best and second-best LLRs
4. Optionally computes intermediate gap proxies (`gap_proxy_2`, `gap_proxy_3`, etc.)

### Parameter Validation

The method validates:
- `logical_error_distribution` is provided (required)
- `num_classes_to_explore` is provided and >= 1
- Distribution length equals $2^k$ for $k$ observables

---

## Usage Guide

### Basic Usage

```python
import numpy as np
from ldpc_post_selection import SoftOutputsBpLsdDecoder

# Create decoder
decoder = SoftOutputsBpLsdDecoder(circuit=circuit)
num_observables = decoder.obs_matrix.shape[0]

# Create or load logical error distribution
# Shape: (2^k,) where index i = logical error with bit pattern i
logical_error_distribution = np.array([...])  # From empirical data

# Decode with 'most-likely-first' gap proxy
pred, pred_bp, converge, soft_outputs = decoder.decode(
    syndrome,
    compute_logical_gap_proxy=True,
    logical_gap_proxy_method="most-likely-first",
    num_classes_to_explore=10,  # Explore 10 classes total
    logical_error_distribution=logical_error_distribution,
)

# Access results
gap_proxy = soft_outputs["gap_proxy"]
pred_llr = soft_outputs["pred_llr"]
```

### With Intermediate Gap Proxies

```python
pred, pred_bp, converge, soft_outputs = decoder.decode(
    syndrome,
    compute_logical_gap_proxy=True,
    logical_gap_proxy_method="most-likely-first",
    num_classes_to_explore=10,
    logical_error_distribution=logical_error_distribution,
    compute_all_intermediate_gap_proxies=True,
)

# Access intermediate gap proxies
for i in range(2, 11):
    print(f"Gap proxy after {i} classes: {soft_outputs[f'gap_proxy_{i}']}")
```

---

## Collecting Logical Error Distributions

The `logical_error_distribution` module provides utilities to collect empirical error distributions.

### Using `collect_logical_error_distribution`

```python
from ldpc_post_selection import collect_logical_error_distribution

# Run simulation to collect distribution
distribution, metadata = collect_logical_error_distribution(
    circuit=circuit,
    shots=100000,
    seed=42,
)

print(f"Total shots: {metadata['total_shots']}")
print(f"Logical error rate: {metadata['logical_error_rate']:.4f}")
print(f"Distinct nonzero errors: {metadata['nonzero_errors']}")

# Use distribution for gap proxy computation
pred, _, _, soft_outputs = decoder.decode(
    syndrome,
    compute_logical_gap_proxy=True,
    logical_gap_proxy_method="most-likely-first",
    num_classes_to_explore=10,
    logical_error_distribution=distribution,
)
```

### Fast Collection

For better performance with large shot counts:

```python
from ldpc_post_selection import collect_logical_error_distribution_fast

distribution, metadata = collect_logical_error_distribution_fast(
    circuit=circuit,
    shots=100000,
    seed=42,
)
```

### Helper Functions

```python
from ldpc_post_selection import (
    logical_class_to_index,
    index_to_logical_class,
    normalize_distribution,
)

# Convert between representations
logical_class = np.array([True, False, True])  # k=3
index = logical_class_to_index(logical_class)  # Returns 5 (binary: 101)

recovered = index_to_logical_class(5, 3)  # Returns [True, False, True]

# Normalize counts to probabilities
probabilities = normalize_distribution(distribution)
```

---

## Testing

### Test Coverage

The implementation includes comprehensive tests in two files:

#### `tests/test_decoder.py`

| Test | Description |
|------|-------------|
| `test_logical_gap_proxy_most_likely_first_method` | Basic functionality with custom distribution |
| `test_logical_gap_proxy_most_likely_first_with_intermediate` | Intermediate gap proxy computation |
| `test_logical_gap_proxy_most_likely_first_invalid_params` | Parameter validation errors |
| `test_index_to_logical_class_helper` | Index-to-class conversion correctness |
| `test_get_most_likely_logical_classes` | Class selection based on distribution |

#### `tests/test_logical_error_distribution.py`

| Test Class | Tests |
|------------|-------|
| `TestIndexConversions` | Round-trip conversion, edge cases |
| `TestNormalizeDistribution` | Normalization, zero handling |
| `TestCollectLogicalErrorDistribution` | Basic collection, batching, reproducibility |
| `TestCollectLogicalErrorDistributionFast` | Fast version equivalence |

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_decoder.py -v

# Run specific test
pytest tests/test_decoder.py::TestSoftOutputsBpLsdDecoder::test_logical_gap_proxy_most_likely_first_method -v
```

### Example Test: Class Selection

```python
def test_get_most_likely_logical_classes(self, circuit_data):
    decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])

    best_logical_class = np.array([False, False], dtype=bool)
    # Distribution: index 3 most likely, index 1 second, index 2 third
    distribution = np.array([0.1, 0.3, 0.2, 0.4])

    classes = decoder._get_most_likely_logical_classes(
        best_logical_class=best_logical_class,
        logical_error_distribution=distribution,
        num_classes_to_explore=3,
    )

    # First class: [False,False] XOR [True,True] (index 3) = [True,True]
    assert np.array_equal(classes[0], np.array([True, True]))

    # Second class: [False,False] XOR [True,False] (index 1) = [True,False]
    assert np.array_equal(classes[1], np.array([True, False]))
```

---

## Performance Considerations

### Time Complexity

| Method | Classes Explored | Decodings Required |
|--------|------------------|-------------------|
| Exhaustive (`None`) | $2^k$ | $2^k - 1$ |
| Nearby (`'nearby'`) | Variable | Depends on landscape |
| Random (`'random'`) | $n$ | $n - 1$ |
| Most-Likely-First | $n$ | $n - 1$ |

The `'most-likely-first'` method has the same asymptotic complexity as `'random'`, but with better constant factors due to:
- Simple sorting operation (O($2^k \log 2^k$)) done once per decode
- No random number generation during exploration

### Memory Usage

- Distribution array: $O(2^k)$ floats
- This limits practical use to codes with $k \lesssim 20-25$ logical qubits

### When to Use

**Use `'most-likely-first'` when:**
- You have access to empirical logical error statistics
- The error distribution is significantly non-uniform
- You need better gap proxy estimates with limited exploration budget

**Use `'random'` when:**
- No prior knowledge of error distribution is available
- The distribution is approximately uniform
- You want simpler, prior-free operation

---

## Comparison with Other Methods

### Accuracy vs. Exploration Budget

For a fixed exploration budget $n$:

| Scenario | Best Method |
|----------|-------------|
| Uniform error distribution | `'random'` (equivalent to `'most-likely-first'`) |
| Highly skewed distribution | `'most-likely-first'` |
| Very small budget ($n \ll k$) | `'most-likely-first'` |
| Large budget ($n \approx 2^k$) | Exhaustive (`None`) |

### Conceptual Comparison

| Method | Strategy | Pros | Cons |
|--------|----------|------|------|
| Exhaustive | Explore all | Exact gap | Exponential cost |
| Nearby | BFS from best | Efficient for local minima | May miss distant competitors |
| Random | Uniform sampling | No prior needed | Wastes budget on unlikely classes |
| Most-Likely-First | Prioritized by prior | Focuses on likely competitors | Requires prior distribution |

---

## Future Directions

Potential enhancements:

1. **Adaptive distribution updates**: Update the prior distribution during decoding based on observed patterns
2. **Conditional distributions**: Use distributions conditioned on syndrome features (e.g., detector density)
3. **Hybrid methods**: Combine `'most-likely-first'` with `'nearby'` for broader coverage
4. **Distribution compression**: Efficiently represent distributions for large $k$

---

## References

- Original post-selection paper: ["Efficient Post-Selection for General Quantum LDPC Codes"](https://arxiv.org/abs/2510.05795)
- BP+LSD decoder: `ldpc` package documentation
- This implementation: `src/ldpc_post_selection/bplsd_decoder.py`
