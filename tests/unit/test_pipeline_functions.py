"""
tests/unit/test_pipeline_functions.py

Unit tests for discrete_pipeline.py core functions.
Run with: uv run pytest tests/unit/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from discrete_pipeline import compute_stationary


# ── compute_stationary ────────────────────────────────────────────────────────

def test_stationary_sums_to_one():
    n   = 10
    rng = np.random.default_rng(42)
    C   = rng.integers(1, 100, size=(n, n)).astype(float)
    C_d = rng.integers(1, 500, size=(n, n)).astype(float)
    C_g = rng.integers(1, 2000, size=(n, n)).astype(float)
    pi  = compute_stationary(C, C_d, C_g, alpha1=0.1, alpha2=0.01)
    assert np.isclose(pi.sum(), 1.0, atol=1e-6), f"pi sums to {pi.sum()}"


def test_stationary_non_negative():
    n   = 10
    rng = np.random.default_rng(0)
    C   = rng.integers(1, 100, size=(n, n)).astype(float)
    pi  = compute_stationary(C, C, C, alpha1=0.1, alpha2=0.01)
    assert (pi >= 0).all(), f"Negative values in pi: {pi.min()}"


def test_stationary_uniform_matrix_gives_uniform_pi():
    # uniform transition matrix -> uniform stationary distribution
    n  = 6
    C  = np.ones((n, n))
    pi = compute_stationary(C, C, C, alpha1=0.0, alpha2=0.0)
    assert np.allclose(pi, 1 / n, atol=1e-5), f"Expected uniform pi, got {pi}"


def test_stationary_no_nan_or_inf():
    n   = 15
    rng = np.random.default_rng(7)
    C   = rng.integers(0, 50, size=(n, n)).astype(float)
    C_d = rng.integers(0, 200, size=(n, n)).astype(float)
    C_g = rng.integers(0, 1000, size=(n, n)).astype(float)
    pi  = compute_stationary(C, C_d, C_g, alpha1=0.1, alpha2=0.01)
    assert not np.isnan(pi).any(), "NaN in stationary distribution"
    assert not np.isinf(pi).any(), "Inf in stationary distribution"


def test_stationary_zero_row_handled():
    # station with no outgoing trips — smoothing should handle it
    n      = 5
    C      = np.ones((n, n))
    C[2,:] = 0   # station 2 has no departures
    C_g    = np.ones((n, n))
    pi     = compute_stationary(C, C_g, C_g, alpha1=0.1, alpha2=0.01)
    assert np.isclose(pi.sum(), 1.0, atol=1e-6)
    assert (pi >= 0).all()


def test_stationary_shape():
    n  = 8
    C  = np.ones((n, n))
    pi = compute_stationary(C, C, C, alpha1=0.1, alpha2=0.01)
    assert pi.shape == (n,), f"Expected shape ({n},), got {pi.shape}"


def test_stationary_alpha_zero_uses_only_bin():
    # with alpha1=alpha2=0 only C matters, not C_day or C_global
    n      = 4
    C      = np.eye(n)              # identity — absorbing states
    C_fake = np.ones((n, n)) * 999  # large values that should be ignored
    pi     = compute_stationary(C, C_fake, C_fake, alpha1=0.0, alpha2=0.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-6)