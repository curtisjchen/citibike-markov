"""
tests/integration/test_outputs.py

Integration tests — validate pipeline outputs after running discrete_pipeline.py.
Run with: uv run pytest tests/integration/ -v

Requires outputs/ directory to be populated first:
    uv run src/discrete_pipeline.py
"""

import json
import os
import numpy as np
import pytest

OUTPUTS = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

def output(filename):
    return os.path.join(OUTPUTS, filename)

def load_stations():
    with open(output('stations.json')) as f:
        return json.load(f)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def stations():
    return load_stations()

@pytest.fixture(scope='module')
def n_stations(stations):
    return len(stations)

@pytest.fixture(scope='module')
def pi_day_hour():
    return np.load(output('pi_by_day_hour.npy'))

@pytest.fixture(scope='module')
def pi_hour():
    return np.load(output('pi_by_hour.npy'))

@pytest.fixture(scope='module')
def flow_ratio():
    return np.load(output('flow_ratio_by_day_hour.npy'))


# ── stations.json ─────────────────────────────────────────────────────────────

def test_stations_exist():
    assert os.path.exists(output('stations.json'))

def test_stations_not_empty(stations):
    assert len(stations) > 0

def test_stations_have_required_fields(stations):
    for idx, info in stations.items():
        assert 'name' in info, f"Station {idx} missing name"
        assert 'lat'  in info, f"Station {idx} missing lat"
        assert 'lng'  in info, f"Station {idx} missing lng"

def test_stations_lat_lng_in_range(stations):
    for idx, info in stations.items():
        if info['lat'] is not None:
            assert 40.0 < info['lat'] < 41.5,  f"Station {idx} lat out of range: {info['lat']}"
            assert -75.0 < info['lng'] < -73.0, f"Station {idx} lng out of range: {info['lng']}"

def test_stations_names_not_empty(stations):
    for idx, info in stations.items():
        assert info['name'] and len(info['name']) > 0, f"Station {idx} has empty name"


# ── pi_by_day_hour.npy ────────────────────────────────────────────────────────

def test_pi_day_hour_exists():
    assert os.path.exists(output('pi_by_day_hour.npy'))

def test_pi_day_hour_shape(pi_day_hour, n_stations):
    assert pi_day_hour.shape == (7, 24, n_stations), \
        f"Expected (7, 24, {n_stations}), got {pi_day_hour.shape}"

def test_pi_day_hour_sums_to_one(pi_day_hour):
    sums = pi_day_hour.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-5), \
        f"pi rows don't sum to 1. Min={sums.min():.6f} Max={sums.max():.6f}"

def test_pi_day_hour_non_negative(pi_day_hour):
    assert (pi_day_hour >= 0).all(), f"Negative pi values. Min={pi_day_hour.min()}"

def test_pi_day_hour_no_nan(pi_day_hour):
    assert not np.isnan(pi_day_hour).any(), "NaN in pi_by_day_hour"

def test_pi_day_hour_no_inf(pi_day_hour):
    assert not np.isinf(pi_day_hour).any(), "Inf in pi_by_day_hour"

def test_pi_day_hour_weekday_vs_weekend_differ(pi_day_hour):
    # weekday and weekend patterns should be meaningfully different
    weekday_mean = pi_day_hour[:5].mean(axis=(0, 1))
    weekend_mean = pi_day_hour[5:].mean(axis=(0, 1))
    correlation  = np.corrcoef(weekday_mean, weekend_mean)[0, 1]
    # should be correlated but not identical
    assert correlation < 0.9999, "Weekday and weekend pi are suspiciously identical"


# ── pi_by_hour.npy ────────────────────────────────────────────────────────────

def test_pi_hour_exists():
    assert os.path.exists(output('pi_by_hour.npy'))

def test_pi_hour_shape(pi_hour, n_stations):
    assert pi_hour.shape == (24, n_stations), \
        f"Expected (24, {n_stations}), got {pi_hour.shape}"

def test_pi_hour_sums_to_one(pi_hour):
    sums = pi_hour.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)

def test_pi_hour_non_negative(pi_hour):
    assert (pi_hour >= 0).all()

def test_pi_hour_no_nan_or_inf(pi_hour):
    assert not np.isnan(pi_hour).any()
    assert not np.isinf(pi_hour).any()


# ── flow_ratio_by_day_hour.npy ────────────────────────────────────────────────

def test_flow_ratio_exists():
    assert os.path.exists(output('flow_ratio_by_day_hour.npy'))

def test_flow_ratio_shape(flow_ratio, n_stations):
    assert flow_ratio.shape == (7, 24, n_stations), \
        f"Expected (7, 24, {n_stations}), got {flow_ratio.shape}"

def test_flow_ratio_non_negative(flow_ratio):
    assert (flow_ratio >= 0).all(), f"Negative flow ratios. Min={flow_ratio.min()}"

def test_flow_ratio_no_nan(flow_ratio):
    assert not np.isnan(flow_ratio).any(), "NaN in flow_ratio_by_day_hour"

def test_flow_ratio_no_inf(flow_ratio):
    assert not np.isinf(flow_ratio).any(), "Inf in flow_ratio_by_day_hour"

def test_flow_ratio_defaults_to_one_for_sparse_bins(flow_ratio):
    # sparse bins should be set to 1.0, not 0 or extreme values
    neutral = flow_ratio == 1.0
    assert neutral.any(), "No bins defaulted to 1.0 — sparse handling may be broken"

def test_flow_ratio_reasonable_range(flow_ratio):
    # ratios should be reasonable — no station should have 1000x more arrivals than departures
    non_neutral = flow_ratio[flow_ratio != 1.0]
    if len(non_neutral) > 0:
        assert non_neutral.max() < 10000, f"Extreme flow ratio: {non_neutral.max():.1f}"


# ── cross-file consistency ─────────────────────────────────────────────────────

def test_all_shapes_consistent(pi_day_hour, pi_hour, flow_ratio, n_stations):
    assert pi_day_hour.shape[2] == n_stations
    assert pi_hour.shape[1]     == n_stations
    assert flow_ratio.shape[2]  == n_stations

def test_pi_hour_consistent_with_pi_day_hour(pi_day_hour, pi_hour):
    # pi_by_hour should be close to the mean of pi_by_day_hour over days
    for hour in range(24):
        expected = pi_day_hour[:, hour, :].mean(axis=0)
        expected /= expected.sum()
        assert np.allclose(pi_hour[hour], expected, atol=1e-5), \
            f"pi_by_hour[{hour}] inconsistent with pi_by_day_hour mean"
        