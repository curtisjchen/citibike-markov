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


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def bin_config():
    return np.load(output('bin_config.npy'))

@pytest.fixture(scope='module')
def bin_minutes(bin_config):
    return int(bin_config[0])

@pytest.fixture(scope='module')
def n_bins(bin_config):
    return int(bin_config[1])

@pytest.fixture(scope='module')
def stations():
    with open(output('stations.json')) as f:
        return json.load(f)

@pytest.fixture(scope='module')
def n_stations(stations):
    return len(stations)

@pytest.fixture(scope='module')
def pi_day_bin():
    return np.load(output('pi_by_day_bin.npy'))

@pytest.fixture(scope='module')
def pi_bin():
    return np.load(output('pi_by_bin.npy'))

@pytest.fixture(scope='module')
def flow_ratio():
    return np.load(output('flow_ratio_by_day_bin.npy'))


# ── bin_config.npy ────────────────────────────────────────────────────────────

def test_bin_config_exists():
    assert os.path.exists(output('bin_config.npy'))

def test_bin_config_valid_bin_minutes(bin_minutes):
    assert bin_minutes in (15, 30, 60), f"Unexpected bin_minutes: {bin_minutes}"

def test_bin_config_n_bins_consistent(bin_minutes, n_bins):
    expected = 60 * 24 // bin_minutes
    assert n_bins == expected, f"Expected {expected} bins for {bin_minutes}min, got {n_bins}"


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


# ── pi_by_day_bin.npy ─────────────────────────────────────────────────────────

def test_pi_day_bin_exists():
    assert os.path.exists(output('pi_by_day_bin.npy'))

def test_pi_day_bin_shape(pi_day_bin, n_bins, n_stations):
    assert pi_day_bin.shape == (7, n_bins, n_stations), \
        f"Expected (7, {n_bins}, {n_stations}), got {pi_day_bin.shape}"

def test_pi_day_bin_sums_to_one(pi_day_bin):
    sums = pi_day_bin.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-5), \
        f"pi rows don't sum to 1. Min={sums.min():.6f} Max={sums.max():.6f}"

def test_pi_day_bin_non_negative(pi_day_bin):
    assert (pi_day_bin >= 0).all(), f"Negative pi values. Min={pi_day_bin.min()}"

def test_pi_day_bin_no_nan(pi_day_bin):
    assert not np.isnan(pi_day_bin).any(), "NaN in pi_by_day_bin"

def test_pi_day_bin_no_inf(pi_day_bin):
    assert not np.isinf(pi_day_bin).any(), "Inf in pi_by_day_bin"

def test_pi_day_bin_weekday_vs_weekend_differ(pi_day_bin):
    weekday_mean = pi_day_bin[:5].mean(axis=(0, 1))
    weekend_mean = pi_day_bin[5:].mean(axis=(0, 1))
    correlation  = np.corrcoef(weekday_mean, weekend_mean)[0, 1]
    assert correlation < 0.9999, "Weekday and weekend pi are suspiciously identical"


# ── pi_by_bin.npy ─────────────────────────────────────────────────────────────

def test_pi_bin_exists():
    assert os.path.exists(output('pi_by_bin.npy'))

def test_pi_bin_shape(pi_bin, n_bins, n_stations):
    assert pi_bin.shape == (n_bins, n_stations), \
        f"Expected ({n_bins}, {n_stations}), got {pi_bin.shape}"

def test_pi_bin_sums_to_one(pi_bin):
    sums = pi_bin.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)

def test_pi_bin_non_negative(pi_bin):
    assert (pi_bin >= 0).all()

def test_pi_bin_no_nan_or_inf(pi_bin):
    assert not np.isnan(pi_bin).any()
    assert not np.isinf(pi_bin).any()


# ── flow_ratio_by_day_bin.npy ─────────────────────────────────────────────────

def test_flow_ratio_exists():
    assert os.path.exists(output('flow_ratio_by_day_bin.npy'))

def test_flow_ratio_shape(flow_ratio, n_bins, n_stations):
    assert flow_ratio.shape == (7, n_bins, n_stations), \
        f"Expected (7, {n_bins}, {n_stations}), got {flow_ratio.shape}"

def test_flow_ratio_non_negative(flow_ratio):
    assert (flow_ratio >= 0).all(), f"Negative flow ratios. Min={flow_ratio.min()}"

def test_flow_ratio_no_nan(flow_ratio):
    assert not np.isnan(flow_ratio).any(), "NaN in flow_ratio_by_day_bin"

def test_flow_ratio_no_inf(flow_ratio):
    assert not np.isinf(flow_ratio).any(), "Inf in flow_ratio_by_day_bin"

def test_flow_ratio_defaults_to_one_for_sparse_bins(flow_ratio):
    assert (flow_ratio == 1.0).any(), "No bins defaulted to 1.0 — sparse handling may be broken"

def test_flow_ratio_reasonable_range(flow_ratio):
    non_neutral = flow_ratio[flow_ratio != 1.0]
    if len(non_neutral) > 0:
        assert non_neutral.max() < 100, f"Extreme flow ratio: {non_neutral.max():.1f}"


# ── cross-file consistency ────────────────────────────────────────────────────

def test_all_shapes_consistent(pi_day_bin, pi_bin, flow_ratio, n_bins, n_stations):
    assert pi_day_bin.shape  == (7, n_bins, n_stations)
    assert pi_bin.shape      == (n_bins, n_stations)
    assert flow_ratio.shape  == (7, n_bins, n_stations)

def test_pi_bin_consistent_with_pi_day_bin(pi_day_bin, pi_bin):
    for b in range(pi_day_bin.shape[1]):
        expected = pi_day_bin[:, b, :].mean(axis=0)
        expected /= expected.sum()
        assert np.allclose(pi_bin[b], expected, atol=1e-5), \
            f"pi_by_bin[{b}] inconsistent with pi_by_day_bin mean"