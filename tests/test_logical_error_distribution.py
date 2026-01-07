"""Tests for the logical_error_distribution module."""

import numpy as np
import pytest
import stim

from ldpc_post_selection.logical_error_distribution import (
    collect_logical_error_distribution,
    collect_logical_error_distribution_fast,
    index_to_logical_class,
    logical_class_to_index,
    normalize_distribution,
)


class TestIndexConversions:
    """Tests for index to logical class conversion functions."""

    def test_logical_class_to_index_basic(self):
        """Test basic logical class to index conversion."""
        # All False -> 0
        assert logical_class_to_index(np.array([False, False, False])) == 0

        # First bit True -> 1
        assert logical_class_to_index(np.array([True, False, False])) == 1

        # Second bit True -> 2
        assert logical_class_to_index(np.array([False, True, False])) == 2

        # First and second bits True -> 3
        assert logical_class_to_index(np.array([True, True, False])) == 3

        # Third bit True -> 4
        assert logical_class_to_index(np.array([False, False, True])) == 4

        # All True -> 7
        assert logical_class_to_index(np.array([True, True, True])) == 7

    def test_index_to_logical_class_basic(self):
        """Test basic index to logical class conversion."""
        # 0 -> all False
        result = index_to_logical_class(0, 3)
        assert np.array_equal(result, np.array([False, False, False]))

        # 1 -> first bit True
        result = index_to_logical_class(1, 3)
        assert np.array_equal(result, np.array([True, False, False]))

        # 5 -> first and third bits True (binary: 101)
        result = index_to_logical_class(5, 3)
        assert np.array_equal(result, np.array([True, False, True]))

        # 7 -> all True
        result = index_to_logical_class(7, 3)
        assert np.array_equal(result, np.array([True, True, True]))

    def test_round_trip_conversion(self):
        """Test that conversion is invertible."""
        for num_obs in [1, 2, 3, 4, 5]:
            for idx in range(1 << num_obs):
                logical_class = index_to_logical_class(idx, num_obs)
                recovered_idx = logical_class_to_index(logical_class)
                assert recovered_idx == idx, f"Failed for idx={idx}, num_obs={num_obs}"

    def test_empty_logical_class(self):
        """Test conversion with empty logical class (0 observables)."""
        assert logical_class_to_index(np.array([], dtype=bool)) == 0
        result = index_to_logical_class(0, 0)
        assert len(result) == 0


class TestNormalizeDistribution:
    """Tests for the normalize_distribution function."""

    def test_normalize_counts(self):
        """Test normalizing count distribution."""
        counts = np.array([10, 20, 30, 40])
        normalized = normalize_distribution(counts)

        assert np.isclose(normalized.sum(), 1.0)
        assert np.allclose(normalized, [0.1, 0.2, 0.3, 0.4])

    def test_normalize_zeros(self):
        """Test normalizing all-zero distribution returns uniform."""
        zeros = np.zeros(4)
        normalized = normalize_distribution(zeros)

        assert np.isclose(normalized.sum(), 1.0)
        assert np.allclose(normalized, [0.25, 0.25, 0.25, 0.25])

    def test_normalize_single_element(self):
        """Test normalizing single non-zero element."""
        single = np.array([0, 0, 5, 0])
        normalized = normalize_distribution(single)

        assert np.isclose(normalized.sum(), 1.0)
        assert np.isclose(normalized[2], 1.0)


class TestCollectLogicalErrorDistribution:
    """Tests for the collect_logical_error_distribution function."""

    @pytest.fixture
    def simple_circuit(self):
        """Create a simple repetition code circuit."""
        return stim.Circuit.generated(
            "repetition_code:memory",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )

    @pytest.fixture
    def surface_code_circuit(self):
        """Create a surface code circuit."""
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )

    def test_basic_collection(self, surface_code_circuit):
        """Test basic logical error distribution collection."""
        distribution, metadata = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=100,
            seed=42,
        )

        # Check distribution shape
        num_obs = metadata["num_observables"]
        assert distribution.shape == (1 << num_obs,)

        # Check distribution sums to total shots
        assert distribution.sum() == metadata["total_shots"]

        # Check metadata
        assert metadata["total_shots"] == 100
        assert metadata["num_observables"] >= 1
        assert 0.0 <= metadata["logical_error_rate"] <= 1.0

    def test_distribution_values_non_negative(self, surface_code_circuit):
        """Test that all distribution values are non-negative."""
        distribution, _ = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=50,
            seed=42,
        )

        assert np.all(distribution >= 0)

    def test_batch_processing(self, surface_code_circuit):
        """Test that batch processing produces valid results."""
        # Run with batching
        dist_batched, meta_batched = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=100,
            batch_size=25,
            seed=42,
        )

        # Verify batched results are valid
        assert dist_batched.sum() == meta_batched["total_shots"]
        assert meta_batched["total_shots"] == 100
        assert np.all(dist_batched >= 0)

        # Verify metadata is consistent
        assert 0.0 <= meta_batched["logical_error_rate"] <= 1.0
        assert meta_batched["num_observables"] >= 1

    def test_reproducibility_with_seed(self, surface_code_circuit):
        """Test that results are reproducible with the same seed."""
        dist1, _ = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=50,
            seed=12345,
        )

        dist2, _ = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=50,
            seed=12345,
        )

        assert np.array_equal(dist1, dist2)

    def test_no_observables_raises_error(self):
        """Test that circuit without observables raises error."""
        # Create circuit without observables
        circuit = stim.Circuit()
        circuit.append_operation("R", [0, 1])
        circuit.append_operation("MR", [0, 1])
        circuit.append_operation("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

        with pytest.raises(ValueError, match="no observables"):
            collect_logical_error_distribution(circuit=circuit, shots=10)


class TestCollectLogicalErrorDistributionFast:
    """Tests for the fast version of collect_logical_error_distribution."""

    @pytest.fixture
    def surface_code_circuit(self):
        """Create a surface code circuit."""
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )

    def test_fast_collection(self, surface_code_circuit):
        """Test fast logical error distribution collection."""
        distribution, metadata = collect_logical_error_distribution_fast(
            circuit=surface_code_circuit,
            shots=100,
            seed=42,
        )

        # Check distribution shape
        num_obs = metadata["num_observables"]
        assert distribution.shape == (1 << num_obs,)

        # Check distribution sums to total shots
        assert distribution.sum() == metadata["total_shots"]

    def test_fast_matches_regular(self, surface_code_circuit):
        """Test that fast version produces same results as regular version."""
        dist_regular, meta_regular = collect_logical_error_distribution(
            circuit=surface_code_circuit,
            shots=50,
            seed=42,
        )

        dist_fast, meta_fast = collect_logical_error_distribution_fast(
            circuit=surface_code_circuit,
            shots=50,
            seed=42,
        )

        assert np.array_equal(dist_regular, dist_fast)
        assert meta_regular["num_observables"] == meta_fast["num_observables"]
        assert meta_regular["total_shots"] == meta_fast["total_shots"]
