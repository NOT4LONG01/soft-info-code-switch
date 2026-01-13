# Logical Gap Proxy Methods for Quantum Error Correction

This document provides a high-level overview of logical gap proxy computation methods used in quantum error correction decoding. The logical gap proxy serves as a confidence metric that estimates how certain a decoder is about its prediction.

## Table of Contents

1. [Theoretical Motivation](#theoretical-motivation)
2. [The Logical Gap](#the-logical-gap)
3. [Gap Proxy Methods](#gap-proxy-methods)
   - [Exhaustive Exploration](#exhaustive-exploration)
   - [Nearby Exploration](#nearby-exploration)
   - [Random Sampling](#random-sampling)
   - [Most-Likely-First](#most-likely-first)
   - [Weighted-Random Sampling](#weighted-random-sampling)
   - [Adaptive Methods](#adaptive-methods)
4. [Collecting Logical Error Distributions](#collecting-logical-error-distributions)
5. [Summary](#summary)

---

## Theoretical Motivation

In quantum error correction, a decoder receives syndrome information (parity check violations) and must predict which physical errors occurred. For codes encoding multiple logical qubits, there are multiple **logical classes**—equivalence classes of physical errors that differ by logical operators.

A decoder's prediction can be:
- **Correct**: The predicted error and true error are in the same logical class
- **Incorrect**: They differ by a logical error (undetected failure)

The challenge is that a decoder cannot know with certainty whether its prediction is correct. The **logical gap** provides a measure of confidence: a large gap indicates the decoder's chosen logical class is significantly more likely than alternatives, while a small gap suggests competing classes have similar likelihoods.

### Post-Selection Application

The logical gap enables **post-selection**—discarding decoding results with low confidence. By filtering out low-gap predictions, one can achieve lower effective logical error rates at the cost of reduced throughput. This trade-off is valuable in scenarios where:
- Verification costs are high
- A small fraction of high-confidence results is acceptable
- Error rates must be suppressed below what standard decoding achieves

---

## The Logical Gap

### Definition

For a code with $k$ logical qubits, there are $2^k$ logical classes. Given syndrome $s$, a decoder finds the most likely error pattern for each logical class $c$, yielding a **prediction weight** (or log-likelihood ratio) $\text{LLR}(c)$.

Let $c^*$ denote the logical class with the minimum LLR (the decoder's prediction). The **logical gap** is:

$$\text{gap} = \min_{c \neq c^*} \text{LLR}(c) - \text{LLR}(c^*)$$

A larger gap means the best prediction is significantly better than all alternatives.

### Computational Challenge

Computing the exact logical gap requires:
1. Running the decoder for each of the $2^k$ logical classes
2. Finding the minimum and second-minimum LLRs

For codes with many logical qubits (e.g., $k > 10$), this becomes computationally prohibitive. **Gap proxy methods** approximate the gap by exploring only a subset of logical classes.

---

## Gap Proxy Methods

### Exhaustive Exploration

**Overview**: Explore all $2^k$ logical classes to compute the exact gap.

**Pseudocode**:
```
function EXHAUSTIVE_GAP(syndrome, initial_class, k):
    // Input: syndrome s, initial best class c*, number of observables k
    // Output: exact logical gap

    explored ← empty dictionary
    explored[initial_class] ← DECODE_FIXED_CLASS(syndrome, initial_class)

    // Explore all 2^k logical classes
    for c in {0,1}^k:
        if c ≠ initial_class:
            explored[c] ← DECODE_FIXED_CLASS(syndrome, c)

    // Find minimum and second-minimum LLRs
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1]

    return second_best_llr - best_llr
```

**Characteristics**:
- Exact: Returns the true logical gap
- Complexity: $O(2^k)$ decoder invocations
- Practical limit: Feasible for $k \lesssim 8-10$

### Nearby Exploration

**Overview**: Iteratively explore logical classes that differ from the current best by single bit flips (Hamming distance 1), using a breadth-first search strategy.

**Pseudocode**:
```
function NEARBY_GAP(syndrome, initial_class, k):
    // Input: syndrome s, initial best class c*, number of observables k
    // Output: gap proxy from local exploration

    explored ← empty dictionary
    explored_set ← {initial_class}
    explored[initial_class] ← DECODE_FIXED_CLASS(syndrome, initial_class)

    queue ← [initial_class]

    while queue is not empty:
        current_class ← queue.pop_front()
        current_best_llr ← min(llr for (llr, _) in explored.values())

        // Explore all k neighbors (Hamming distance 1)
        for i in 0 to k-1:
            neighbor ← FLIP_BIT(current_class, i)
            if neighbor not in explored_set:
                explored_set.add(neighbor)
                llr ← DECODE_FIXED_CLASS(syndrome, neighbor)
                explored[neighbor] ← llr

                // If better than current best, add to queue for further exploration
                if llr < current_best_llr:
                    queue.append(neighbor)
                    current_best_llr ← llr

    // Compute gap from all explored classes
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

**Characteristics**:
- Adaptive: Follows the "gradient" toward lower LLR
- Complexity: Variable, depends on the LLR landscape
- Best for: Scenarios where the optimal class is near the initial prediction
- Weakness: May miss distant competitors if the LLR landscape has multiple local minima

### Random Sampling

**Overview**: Uniformly sample a fixed number of logical classes to estimate the gap.

**Pseudocode**:
```
function RANDOM_GAP(syndrome, initial_class, k, n):
    // Input: syndrome s, initial best class c*, number of observables k,
    //        total classes to explore n
    // Output: gap proxy from uniform random sampling

    explored ← empty dictionary
    explored[initial_class] ← DECODE_FIXED_CLASS(syndrome, initial_class)

    // Convert initial_class to integer representation
    initial_int ← BITS_TO_INT(initial_class)

    // Sample n-1 distinct classes uniformly (excluding initial_class)
    sampled_ints ← SAMPLE_WITHOUT_REPLACEMENT(
        range(0, 2^k) \ {initial_int},
        count = min(n-1, 2^k - 1)
    )

    // Decode each sampled class
    for class_int in sampled_ints:
        class_bits ← INT_TO_BITS(class_int, k)
        explored[class_bits] ← DECODE_FIXED_CLASS(syndrome, class_bits)

    // Compute gap proxy
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

**Characteristics**:
- Simple: No prior knowledge required
- Complexity: Exactly $n$ decoder invocations
- Unbiased: Each non-best class has equal probability of being explored
- Weakness: May waste exploration budget on unlikely competitors

#### Coverage-Restricted Random Sampling

When a `coverage_fraction` parameter is specified along with `logical_error_distribution`, the random sampling can be restricted to the most likely logical errors. This combines the simplicity of uniform random sampling with the efficiency of distribution-guided selection.

**Behavior**: Given `coverage_fraction = f`:
1. Exclude the identity error (index 0, representing no logical error)
2. Sort remaining logical errors by probability (descending)
3. Normalize probabilities over non-identity errors only (conditional distribution given a logical error occurred)
4. Compute cumulative probabilities
5. Include only errors where cumulative probability <= f (max cumulative mass not exceeding f)
6. Sample uniformly from this restricted pool

**Important Notes**:
- The normalization excludes the identity error, so `coverage_fraction` represents coverage over the *conditional* distribution P(error | error occurred), not the raw unconditional distribution.
- The eligibility criterion is cumulative probability **<=** f, meaning the total mass of eligible errors will generally be less than f (excluding the first error that would push cumulative over f).

**Example**: With `coverage_fraction = 0.3`:
- If non-identity errors are sorted as [E1: 20%, E2: 15%, E3: 10%, E4: 5%, ...] (normalized)
- Cumulative: [20%, 35%, 45%, 50%, ...]
- Only E1 (cumulative 20% <= 30%) is eligible for sampling
- E2 would have cumulative 35% > 30%, so it is excluded

**Characteristics**:
- Focused: Concentrates on the most likely competitors
- Configurable: `coverage_fraction` controls the trade-off between focus and diversity
- Requires distribution: Unlike pure random, needs `logical_error_distribution`
- Uniform within pool: Once eligible errors are selected, sampling is uniform
- Strict validation: Raises error if all non-identity weights are zero

**Edge Cases**:
- `coverage_fraction = 1.0` or `None`: Falls back to pure random (no distribution required)
- Very small `coverage_fraction`: At least one error (the most likely) is always included
- `num_classes_to_explore` < eligible pool size + 1: Raises error to ensure all eligible errors are explored

### Most-Likely-First

**Overview**: Use a pre-computed logical error distribution to prioritize exploring classes most likely to compete with the best prediction.

**Background**: In practice, logical errors (transitions between logical classes) are not uniform. Some error patterns occur much more frequently than others. By collecting statistics on which logical errors occur most often, we can intelligently select which classes to explore.

**Pseudocode**:
```
function MOST_LIKELY_FIRST_GAP(syndrome, initial_class, k, n, P):
    // Input: syndrome s, initial best class c*, number of observables k,
    //        total classes to explore n, logical error distribution P[0..2^k-1]
    // Output: gap proxy from distribution-guided deterministic selection

    explored ← empty dictionary
    explored[initial_class] ← DECODE_FIXED_CLASS(syndrome, initial_class)

    // Sort error indices by distribution probability (descending)
    // Exclude index 0 (identity/no error)
    sorted_error_indices ← ARGSORT_DESCENDING(P)
    sorted_error_indices ← FILTER(sorted_error_indices, λi. i ≠ 0)

    // Select top n-1 errors and explore corresponding classes
    num_to_explore ← min(n-1, 2^k - 1)
    for i in 0 to num_to_explore - 1:
        error_idx ← sorted_error_indices[i]
        error_pattern ← INT_TO_BITS(error_idx, k)

        // Compute candidate class: initial_class XOR error_pattern
        candidate_class ← initial_class ⊕ error_pattern
        explored[candidate_class] ← DECODE_FIXED_CLASS(syndrome, candidate_class)

    // Compute gap proxy
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

**Key Insight**: If error pattern $e$ frequently causes logical failures, then class $c^* \oplus e$ is likely to be a strong competitor to $c^*$. By exploring these classes first, we maximize the probability of finding the true second-best class within a limited exploration budget.

**Characteristics**:
- Deterministic: Same distribution always produces same exploration order
- Requires prior: Needs pre-computed logical error distribution
- Best for: Highly skewed error distributions
- Complexity: Exactly $n$ decoder invocations

### Weighted-Random Sampling

**Overview**: Sample logical classes with probabilities proportional to the logical error distribution, combining the benefits of random exploration with distribution-guided selection.

**Pseudocode**:
```
function WEIGHTED_RANDOM_GAP(syndrome, initial_class, k, n, P):
    // Input: syndrome s, initial best class c*, number of observables k,
    //        total classes to explore n, logical error distribution P[0..2^k-1]
    // Output: gap proxy from distribution-guided stochastic sampling

    explored ← empty dictionary
    explored[initial_class] ← DECODE_FIXED_CLASS(syndrome, initial_class)

    // Prepare valid error indices (exclude identity at index 0)
    valid_indices ← [1, 2, ..., 2^k - 1]

    // Normalize weights to probabilities
    weights ← [P[i] for i in valid_indices]
    probabilities ← weights / sum(weights)

    // Sample n-1 errors without replacement, weighted by probabilities
    num_to_sample ← min(n-1, 2^k - 1)
    sampled_error_indices ← WEIGHTED_SAMPLE_WITHOUT_REPLACEMENT(
        valid_indices, probabilities, count = num_to_sample
    )

    // Explore corresponding classes
    for error_idx in sampled_error_indices:
        error_pattern ← INT_TO_BITS(error_idx, k)

        // Compute candidate class: initial_class XOR error_pattern
        candidate_class ← initial_class ⊕ error_pattern
        explored[candidate_class] ← DECODE_FIXED_CLASS(syndrome, candidate_class)

    // Compute gap proxy
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

**Comparison with Other Methods**:

| Aspect | Random | Most-Likely-First | Weighted-Random |
|--------|--------|-------------------|-----------------|
| Selection | Uniform | Deterministic top-$n$ | Probability-weighted |
| Uses distribution | No | Yes | Yes |
| Deterministic | No | Yes | No |
| Exploration diversity | High | Low | Medium |

**Characteristics**:
- Stochastic: Different runs may explore different classes
- Biased toward likely errors while maintaining exploration diversity
- Useful when the distribution is approximate or when randomization is desired

### Adaptive Methods

**Overview**: The adaptive variants (`most-likely-first-adaptive` and `weighted-random-adaptive`) dynamically update the base class for selecting candidates whenever a better class is found during exploration.

**Motivation**: Non-adaptive methods compute all candidate classes upfront based on the initial best class $c^*$. If during exploration we find a class $c'$ with lower LLR than $c^*$, subsequent exploration should "recenter" around $c'$ because:
- Logical errors are typically correlated with the current decoding state
- Errors relative to the new best class $c'$ are more informative than errors relative to the original $c^*$

#### Most-Likely-First-Adaptive

**Pseudocode**:
```
function MLF_ADAPTIVE_GAP(syndrome, initial_class, k, n, P):
    // Input: syndrome s, initial best class c*, number of observables k,
    //        total classes to explore n, logical error distribution P[0..2^k-1]
    // Output: gap proxy from adaptive distribution-guided selection

    explored ← empty dictionary
    explored_set ← {initial_class}
    initial_llr ← DECODE_FIXED_CLASS(syndrome, initial_class)
    explored[initial_class] ← initial_llr

    // Pre-sort error indices by distribution probability (descending)
    sorted_error_indices ← ARGSORT_DESCENDING(P)

    // Track current best class and its LLR
    current_best_class ← initial_class
    current_best_llr ← initial_llr

    // Cursor for efficient scanning (reset when best class changes)
    search_cursor ← 0

    while |explored_set| < n:
        // Find next unexplored class by walking through sorted errors
        found ← false
        while search_cursor < |sorted_error_indices| and not found:
            error_idx ← sorted_error_indices[search_cursor]
            search_cursor ← search_cursor + 1

            if error_idx = 0:  // Skip identity
                continue

            error_pattern ← INT_TO_BITS(error_idx, k)
            candidate_class ← current_best_class ⊕ error_pattern

            if candidate_class not in explored_set:
                found ← true

        if not found:
            break  // No more unexplored classes available

        // Decode the candidate class
        llr ← DECODE_FIXED_CLASS(syndrome, candidate_class)
        explored[candidate_class] ← llr
        explored_set.add(candidate_class)

        // Update current best if this class is better
        if llr < current_best_llr:
            current_best_class ← candidate_class
            current_best_llr ← llr
            search_cursor ← 0  // Reset cursor when best class changes

    // Compute gap proxy
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

#### Weighted-Random-Adaptive

**Pseudocode**:
```
function WR_ADAPTIVE_GAP(syndrome, initial_class, k, n, P):
    // Input: syndrome s, initial best class c*, number of observables k,
    //        total classes to explore n, logical error distribution P[0..2^k-1]
    // Output: gap proxy from adaptive weighted random sampling

    explored ← empty dictionary
    explored_set ← {initial_class}
    initial_llr ← DECODE_FIXED_CLASS(syndrome, initial_class)
    explored[initial_class] ← initial_llr

    // Prepare valid error indices and normalized probabilities
    valid_indices ← [1, 2, ..., 2^k - 1]
    weights ← [P[i] for i in valid_indices]
    probabilities ← weights / sum(weights)

    // Track current best class and its LLR
    current_best_class ← initial_class
    current_best_llr ← initial_llr

    max_retries ← 1000

    while |explored_set| < n:
        // Sample next unexplored class using rejection sampling
        found ← false
        for retry in 1 to max_retries:
            error_idx ← WEIGHTED_SAMPLE(valid_indices, probabilities)
            error_pattern ← INT_TO_BITS(error_idx, k)
            candidate_class ← current_best_class ⊕ error_pattern

            if candidate_class not in explored_set:
                found ← true
                break

        if not found:
            break  // Failed to find unexplored class after max retries

        // Decode the candidate class
        llr ← DECODE_FIXED_CLASS(syndrome, candidate_class)
        explored[candidate_class] ← llr
        explored_set.add(candidate_class)

        // Update current best if this class is better
        if llr < current_best_llr:
            current_best_class ← candidate_class
            current_best_llr ← llr

    // Compute gap proxy
    all_llrs ← sort([llr for (llr, _) in explored.values()])
    best_llr ← all_llrs[0]
    second_best_llr ← all_llrs[1] if len(all_llrs) > 1 else best_llr

    return second_best_llr - best_llr
```

**Key Property**: The explored set $\mathcal{E}$ is never reset—only the selection offset changes when $c_{\text{best}}$ updates. This ensures exactly $n$ unique classes are explored.

**Characteristics**:
- Dynamic: Adapts exploration based on discovered structure
- Potentially more efficient: Focuses exploration around promising regions
- Same complexity as non-adaptive variants: $n$ decoder invocations

---

## Collecting Logical Error Distributions

To use distribution-based methods, you need a logical error distribution $P(e)$ where:
- $P(e)$ is the frequency/probability of logical error $e$ occurring
- Index $e = \sum_{j=0}^{k-1} b_j \cdot 2^j$ encodes the bit pattern $(b_0, b_1, \ldots, b_{k-1})$
- Index 0 represents no logical error (successful decoding)

This distribution can be numerically collected by:
1. Running standard decoding on many samples
2. For each sample, computing the logical error as $e = c_{\text{true}} \oplus c_{\text{predicted}}$
3. Counting occurrences of each error pattern

The distribution captures the decoder's failure modes and enables intelligent prioritization of which logical classes to explore.

---

## Summary

The logical gap proxy is a powerful confidence metric for quantum error correction decoding. While computing the exact gap requires exponential work in the number of logical qubits, several proxy methods enable efficient approximation:

- **Exhaustive**: Exact but expensive
- **Nearby**: Efficient local search
- **Random**: Simple, prior-free baseline
- **Most-Likely-First**: Distribution-guided deterministic selection
- **Weighted-Random**: Distribution-guided stochastic selection
- **Adaptive variants**: Dynamic recentering for improved exploration

The choice of method depends on the number of logical qubits, availability of prior error statistics, and computational budget. For most practical applications with $k > 10$, distribution-based methods provide the best trade-off between accuracy and efficiency.
