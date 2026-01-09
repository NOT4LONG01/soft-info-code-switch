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
4. [Method Comparison](#method-comparison)
5. [Choosing a Method](#choosing-a-method)

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

**Algorithm**:
1. For each logical class $c \in \{0, 1\}^k$, perform fixed-class decoding to obtain $\text{LLR}(c)$
2. Find $\text{LLR}^{(1)}$ (minimum) and $\text{LLR}^{(2)}$ (second minimum)
3. Return $\text{gap} = \text{LLR}^{(2)} - \text{LLR}^{(1)}$

**Characteristics**:
- Exact: Returns the true logical gap
- Complexity: $O(2^k)$ decoder invocations
- Practical limit: Feasible for $k \lesssim 8-10$

### Nearby Exploration

**Overview**: Iteratively explore logical classes that differ from the current best by single bit flips (Hamming distance 1), using a breadth-first search strategy.

**Algorithm**:
1. Start with the decoder's initial prediction class $c^*$
2. Explore all $k$ neighboring classes (flip each bit once)
3. If any neighbor has lower LLR than $c^*$, add it to the exploration queue
4. Repeat from the new best class until no better neighbor is found
5. Compute gap from all explored classes

**Characteristics**:
- Adaptive: Follows the "gradient" toward lower LLR
- Complexity: Variable, depends on the LLR landscape
- Best for: Scenarios where the optimal class is near the initial prediction
- Weakness: May miss distant competitors if the LLR landscape has multiple local minima

### Random Sampling

**Overview**: Uniformly sample a fixed number of logical classes to estimate the gap.

**Algorithm**:
1. Start with the initial best class $c^*$
2. Uniformly sample $n-1$ additional classes (excluding $c^*$)
3. Decode each sampled class to obtain its LLR
4. Compute gap proxy from all $n$ explored classes

**Characteristics**:
- Simple: No prior knowledge required
- Complexity: Exactly $n$ decoder invocations
- Unbiased: Each non-best class has equal probability of being explored
- Weakness: May waste exploration budget on unlikely competitors

### Most-Likely-First

**Overview**: Use a pre-computed logical error distribution to prioritize exploring classes most likely to compete with the best prediction.

**Background**: In practice, logical errors (transitions between logical classes) are not uniform. Some error patterns occur much more frequently than others. By collecting statistics on which logical errors occur most often, we can intelligently select which classes to explore.

**Algorithm**:
1. Obtain a logical error distribution $P(e)$ for each error pattern $e \in \{0, 1\}^k$
   - Index 0 represents no error (identity)
   - Higher values indicate more probable errors
2. Sort non-identity error indices by probability (descending)
3. For the top $n-1$ errors $e_1, e_2, \ldots, e_{n-1}$:
   - Compute candidate class as $c^* \oplus e_i$ (XOR operation)
   - Decode this class
4. Compute gap proxy from all explored classes

**Key Insight**: If error pattern $e$ frequently causes logical failures, then class $c^* \oplus e$ is likely to be a strong competitor to $c^*$. By exploring these classes first, we maximize the probability of finding the true second-best class within a limited exploration budget.

**Characteristics**:
- Deterministic: Same distribution always produces same exploration order
- Requires prior: Needs pre-computed logical error distribution
- Best for: Highly skewed error distributions
- Complexity: Exactly $n$ decoder invocations

### Weighted-Random Sampling

**Overview**: Sample logical classes with probabilities proportional to the logical error distribution, combining the benefits of random exploration with distribution-guided selection.

**Algorithm**:
1. Obtain a logical error distribution $P(e)$
2. Sample $n-1$ error indices without replacement, with probability proportional to $P(e)$ (excluding identity)
3. For each sampled error $e_i$:
   - Compute candidate class as $c^* \oplus e_i$
   - Decode this class
4. Compute gap proxy from all explored classes

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

**Algorithm (Adaptive)**:
1. Initialize: Set current best class $c_{\text{best}} = c^*$, explored set $\mathcal{E} = \{c^*\}$
2. While $|\mathcal{E}| < n$:
   - Select next candidate based on $c_{\text{best}}$ (not original $c^*$):
     - **MLF-Adaptive**: Walk through sorted errors, find first $e$ where $c_{\text{best}} \oplus e \notin \mathcal{E}$
     - **WR-Adaptive**: Sample error $e$ from distribution, reject if $c_{\text{best}} \oplus e \in \mathcal{E}$
   - Decode the candidate class $c'$
   - Add $c'$ to $\mathcal{E}$
   - If $\text{LLR}(c') < \text{LLR}(c_{\text{best}})$: Update $c_{\text{best}} = c'$
3. Compute gap proxy from all explored classes

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
