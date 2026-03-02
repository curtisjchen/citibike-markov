"""
discrete_pipeline.py — Discrete-time Markov Chain pipeline

Loads raw trip CSVs, filters to the strongly connected component,
builds hierarchically smoothed transition matrices, computes stationary
distributions and flow ratios, and saves outputs.

Usage:
    uv run src/discrete_pipeline.py
    uv run src/discrete_pipeline.py --alpha1 0.05 --alpha2 0.005 --min-each 10
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DEFAULT_DECAY      = 0.9995
DEFAULT_MIN_EACH   = 5
DEFAULT_ALPHA1     = 0.1
DEFAULT_ALPHA2     = 0.01
DEFAULT_BIN_MINUTES = 60


def load_and_filter(data_dir, decay, bin_minutes=60):
    print("\n[1/4] Loading trip data...")
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"      Found {len(files)} files")

    df = pd.concat([
        pd.read_csv(f, dtype={
            'start_station_id': 'str',
            'end_station_id':   'str',
            'member_casual':    'category',
            'rideable_type':    'category',
        }, low_memory=False)
        for f in files
    ], ignore_index=True)

    # normalise column names across all Citibike schema versions
    # 2013-2020: space-separated lowercase with lat/lng as "latitude"/"longitude"
    # 2021-2023: snake_case with start_lat/start_lng
    # 2024+:     same snake_case but ride_id instead of bikeid
    COLUMN_MAP = {
        'starttime':                  'started_at',
        'stoptime':                   'ended_at',
        'start station name':         'start_station_id',
        'end station name':           'end_station_id',
        'start station latitude':     'start_lat',
        'start station longitude':    'start_lng',
        'end station latitude':       'end_lat',
        'end station longitude':      'end_lng',
    }
    df = df.rename(columns=COLUMN_MAP)

    # use station name as the primary identifier — stable across schema versions
    # strip whitespace and normalise case to avoid duplicates from minor formatting differences
    df['start_station_id'] = df['start_station_id'].astype(str).str.strip()
    df['end_station_id']   = df['end_station_id'].astype(str).str.strip()

    df = df.dropna(subset=['start_station_id', 'end_station_id', 'started_at'])
    df = df[df['start_station_id'] != 'nan']
    df = df[df['end_station_id']   != 'nan']
    df['started_at']  = pd.to_datetime(df['started_at'])
    df['hour']        = df['started_at'].dt.hour
    df['day_of_week'] = df['started_at'].dt.dayofweek  # 0=Mon, 6=Sun
    df['bin']         = (df['started_at'].dt.hour * 60 + df['started_at'].dt.minute) // bin_minutes

    print(f"      Trips loaded: {len(df):,}")
    print(f"      Date range:   {df['started_at'].min()} -> {df['started_at'].max()}")

    reference_date = df['started_at'].max()
    df['days_ago'] = (reference_date - df['started_at']).dt.days
    df['weight']   = decay ** df['days_ago']
    print(f"      Decay={decay}  1yr={decay**365:.3f}  2yr={decay**730:.3f}")

    print("\n[2/4] Filtering to strongly connected component...")
    all_stations = pd.concat([df['start_station_id'], df['end_station_id']]).unique()
    n_raw        = len(all_stations)
    raw_to_idx   = {s: i for i, s in enumerate(all_stations)}

    C_raw = np.zeros((n_raw, n_raw))
    np.add.at(C_raw, (
        df['start_station_id'].map(raw_to_idx).values,
        df['end_station_id'].map(raw_to_idx).values
    ), df['weight'].values)

    _, labels    = connected_components(csr_matrix(C_raw), directed=True, connection='strong')
    largest      = pd.Series(labels).value_counts().idxmax()
    keep         = {all_stations[i] for i, l in enumerate(labels) if l == largest}

    df = df[df['start_station_id'].isin(keep) & df['end_station_id'].isin(keep)].copy()

    all_stations   = sorted(pd.concat([df['start_station_id'], df['end_station_id']]).unique())
    n              = len(all_stations)
    station_to_idx = {s: i for i, s in enumerate(all_stations)}
    idx_to_station = {i: s for s, i in station_to_idx.items()}

    df['start_idx'] = df['start_station_id'].map(station_to_idx)
    df['end_idx']   = df['end_station_id'].map(station_to_idx)

    print(f"      Stations: {n} kept, {n_raw - len(keep)} dropped")
    print(f"      Trips:    {len(df):,}")

    return df, n, idx_to_station


def build_count_tensors(df, n, bin_minutes=60):
    n_bins = 60 * 24 // bin_minutes
    print(f"\n[3/4] Building count tensors ({n_bins} bins/day @ {bin_minutes}min resolution)...")

    si = df['start_idx'].values
    ei = df['end_idx'].values
    dd = df['day_of_week'].values
    bb = df['bin'].values
    w  = df['weight'].values

    C_global  = np.zeros((n, n))
    C_per_day = np.zeros((7, n, n))
    C_per_bin = np.zeros((7, n_bins, n, n))

    np.add.at(C_global, (si, ei), w)

    for day in range(7):
        mask = dd == day
        np.add.at(C_per_day[day], (si[mask], ei[mask]), w[mask])

    for day in range(7):
        for b in range(n_bins):
            mask = (dd == day) & (bb == b)
            np.add.at(C_per_bin[day, b], (si[mask], ei[mask]), w[mask])
        print(f"      Day {day+1}/7 done")

    return si, ei, dd, bb, w, C_global, C_per_day, C_per_bin, n_bins


def compute_stationary(C, C_day, C_global, alpha1, alpha2):
    C_s       = C + alpha1 * C_day + alpha2 * C_global
    row_sums  = C_s.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P         = C_s / row_sums
    vals, vecs = linalg.eig(P.T)
    idx       = np.argmin(np.abs(vals - 1))
    pi        = vecs[:, idx].real
    pi        = np.abs(pi) / np.abs(pi).sum()
    return pi


def compute_all_pi(C_per_bin, C_per_day, C_global, n, alpha1, alpha2, n_bins=24):
    print(f"      Computing {7 * n_bins} stationary distributions ({n_bins} bins/day)...")
    pi_by_day_bin = np.zeros((7, n_bins, n))
    for day in range(7):
        for b in range(n_bins):
            pi_by_day_bin[day, b] = compute_stationary(
                C_per_bin[day, b], C_per_day[day], C_global, alpha1, alpha2
            )
        print(f"      Day {day+1}/7 done")

    # marginalise over days for bin-only matrix
    pi_by_bin = np.zeros((n_bins, n))
    for b in range(n_bins):
        pi_by_bin[b] = pi_by_day_bin[:, b, :].mean(axis=0)
        pi_by_bin[b] /= pi_by_bin[b].sum()

    return pi_by_day_bin, pi_by_bin


def compute_flow_ratios(si, ei, dd, bb, w, n, min_each, n_bins=24):
    print(f"      Computing flow ratios (7 x {n_bins} x n)...")
    arrivals   = np.zeros((7, n_bins, n))
    departures = np.zeros((7, n_bins, n))

    for d in range(7):
        for b in range(n_bins):
            mask = (dd == d) & (bb == b)
            np.add.at(arrivals[d, b],   ei[mask], w[mask])
            np.add.at(departures[d, b], si[mask], w[mask])

    flow_ratio_by_day_bin = np.where(
        (arrivals >= min_each) & (departures >= min_each),
        arrivals / (departures + 1e-10),
        1.0
    )
    return flow_ratio_by_day_bin


def build_stations_json(df, idx_to_station):
    print("      Building stations.json...")
    start_meta = df.groupby('start_station_id').agg(
        name=('start_station_name', 'first'),
        lat =('start_lat',          'median'),
        lng =('start_lng',          'median'),
    )
    end_meta = df.groupby('end_station_id').agg(
        name=('end_station_name', 'first'),
        lat =('end_lat',          'median'),
        lng =('end_lng',          'median'),
    )
    idx_to_info = {}
    for idx, station_id in idx_to_station.items():
        row = start_meta.loc[station_id] if station_id in start_meta.index else end_meta.loc[station_id]
        idx_to_info[idx] = {
            'name': row['name'],
            'lat':  None if pd.isna(row['lat']) else float(row['lat']),
            'lng':  None if pd.isna(row['lng']) else float(row['lng']),
        }
    return idx_to_info


def main(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR,
         decay=DEFAULT_DECAY, min_each=DEFAULT_MIN_EACH,
         alpha1=DEFAULT_ALPHA1, alpha2=DEFAULT_ALPHA2,
         bin_minutes=DEFAULT_BIN_MINUTES):

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    n_bins = 60 * 24 // bin_minutes
    print(f"\n  Bin size: {bin_minutes} minutes ({n_bins} bins/day, {7*n_bins} total)")

    df, n, idx_to_station = load_and_filter(data_dir, decay, bin_minutes)
    si, ei, dd, bb, w, C_global, C_per_day, C_per_bin, n_bins = build_count_tensors(df, n, bin_minutes)

    print("\n[4/4] Computing outputs...")
    pi_by_day_bin, pi_by_bin = compute_all_pi(C_per_bin, C_per_day, C_global, n, alpha1, alpha2, n_bins)
    flow_ratio_by_day_bin    = compute_flow_ratios(si, ei, dd, bb, w, n, min_each, n_bins)
    idx_to_info              = build_stations_json(df, idx_to_station)

    np.save(os.path.join(output_dir, 'pi_by_day_bin.npy'),         pi_by_day_bin)
    np.save(os.path.join(output_dir, 'pi_by_bin.npy'),             pi_by_bin)
    np.save(os.path.join(output_dir, 'flow_ratio_by_day_bin.npy'), flow_ratio_by_day_bin)
    # save bin config so app knows how to interpret the arrays
    np.save(os.path.join(output_dir, 'bin_config.npy'),            np.array([bin_minutes, n_bins]))
    with open(os.path.join(output_dir, 'stations.json'), 'w') as f:
        json.dump(idx_to_info, f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  pi_by_day_bin:         {pi_by_day_bin.shape}")
    print(f"  pi_by_bin:             {pi_by_bin.shape}")
    print(f"  flow_ratio_by_day_bin: {flow_ratio_by_day_bin.shape}")
    print(f"  bin_minutes:           {bin_minutes} ({n_bins} bins/day)")
    print(f"  stations:               {len(idx_to_info)}")
    print(f"  outputs -> {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discrete-time Markov Chain pipeline")
    parser.add_argument('--data-dir',   default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--decay',      type=float, default=DEFAULT_DECAY)
    parser.add_argument('--min-each',   type=int,   default=DEFAULT_MIN_EACH)
    parser.add_argument('--alpha1',     type=float, default=DEFAULT_ALPHA1)
    parser.add_argument('--alpha2',       type=float, default=DEFAULT_ALPHA2)
    parser.add_argument('--bin-minutes',   type=int,   default=DEFAULT_BIN_MINUTES,
                        help='Bin size in minutes (60=hourly, 30=half-hourly, 15=quarter-hourly)')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.decay, args.min_each, args.alpha1, args.alpha2, args.bin_minutes)