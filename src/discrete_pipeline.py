"""
discrete_pipeline.py — Discrete-time Markov Chain pipeline

Three-pass design using parquet intermediates — memory efficient for large datasets:
  Pass 1: scan all CSVs for station IDs → build SCC-filtered station index
  Pass 2: stream CSVs → filter to kept stations → partition by day → write 7 parquet files
  Pass 3: load one day-parquet at a time → compute pi for each bin → discard

Peak memory: one day's trips (~1-2GB) + one (n,n) matrix (~46MB)

Usage:
    uv run src/discrete_pipeline.py
    uv run src/discrete_pipeline.py --bin-minutes 30
    uv run src/discrete_pipeline.py --bin-minutes 15
"""

import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DEFAULT_OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DEFAULT_TEMP_DIR    = os.path.join(os.path.dirname(__file__), '..', 'temp')
DEFAULT_DECAY       = 0.9995
DEFAULT_MIN_EACH    = 5
DEFAULT_ALPHA1      = 0.1
DEFAULT_ALPHA2      = 0.01
DEFAULT_BIN_MINUTES = 60

COLUMN_MAP = {
    'starttime':               'started_at',
    'stoptime':                'ended_at',
    'start station id':        'start_station_id',
    'end station id':          'end_station_id',
    'start station name':      'start_station_name',
    'end station name':        'end_station_name',
    'start station latitude':  'start_lat',
    'start station longitude': 'start_lng',
    'end station latitude':    'end_lat',
    'end station longitude':   'end_lng',
}

READ_DTYPE = {
    'start_station_id': 'str',
    'end_station_id':   'str',
}

# columns we need to keep after loading
KEEP_COLS = [
    'started_at', 'ended_at', 'start_station_id', 'end_station_id',
    'start_station_name', 'end_station_name',
    'start_lat', 'start_lng', 'end_lat', 'end_lng',
]


def read_file(f):
    """Read a single CSV, normalise column names, return cleaned df."""
    try:
        df = pd.read_csv(f, dtype=READ_DTYPE, low_memory=False)
    except Exception as e:
        print(f"      Warning: could not read {f}: {e}")
        return pd.DataFrame()

    df = df.rename(columns=COLUMN_MAP)

    if 'start_station_id' not in df.columns or 'started_at' not in df.columns:
        return pd.DataFrame()

    df['start_station_id'] = df['start_station_id'].astype(str).str.strip()
    df['end_station_id']   = df['end_station_id'].astype(str).str.strip()

    df = df.dropna(subset=['start_station_id', 'end_station_id', 'started_at'])
    df = df[df['start_station_id'] != 'nan']
    df = df[df['end_station_id']   != 'nan']

    # keep only columns we need
    present = [c for c in KEEP_COLS if c in df.columns]
    return df[present].copy()


# ── pass 1: build station index ───────────────────────────────────────────────
def build_station_index(files):
    print("\n[1/3] Pass 1 — building station index...")

    # find reference date and collect all station IDs in one scan
    all_ids        = set()
    reference_date = pd.Timestamp.min
    station_meta   = {}   # station_id -> {name, lat, lng}

    for i, f in enumerate(files):
        df = read_file(f)
        if df.empty:
            continue

        dates = pd.to_datetime(df['started_at'], errors='coerce')
        mx    = dates.max()
        if pd.notna(mx) and mx > reference_date:
            reference_date = mx

        all_ids.update(df['start_station_id'].unique())
        all_ids.update(df['end_station_id'].unique())

        # collect metadata
        for col_id, col_name, col_lat, col_lng in [
            ('start_station_id', 'start_station_name', 'start_lat', 'start_lng'),
            ('end_station_id',   'end_station_name',   'end_lat',   'end_lng'),
        ]:
            if col_name in df.columns and col_lat in df.columns:
                meta = df.groupby(col_id).agg(
                    name=(col_name, 'first'),
                    lat =(col_lat,  'median'),
                    lng =(col_lng,  'median'),
                )
                for sid, row in meta.iterrows():
                    if sid not in station_meta:
                        station_meta[sid] = {
                            'name': row['name'],
                            'lat':  row['lat'],
                            'lng':  row['lng'],
                        }

        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(files)} files scanned, {len(all_ids)} stations")

    print(f"      Reference date: {reference_date.date()}")
    print(f"      Stations before SCC: {len(all_ids)}")

    # build connectivity matrix for SCC filter
    all_stations = sorted(all_ids)
    n_raw        = len(all_stations)
    raw_to_idx   = {s: i for i, s in enumerate(all_stations)}

    C_raw = np.zeros((n_raw, n_raw), dtype='float32')
    for f in files:
        df = read_file(f)
        if df.empty:
            continue
        df = df[df['start_station_id'].isin(raw_to_idx)]
        df = df[df['end_station_id'].isin(raw_to_idx)]
        if df.empty:
            continue
        si = df['start_station_id'].map(raw_to_idx).values.astype('int32')
        ei = df['end_station_id'].map(raw_to_idx).values.astype('int32')
        np.add.at(C_raw, (si, ei), 1)

    _, labels = connected_components(csr_matrix(C_raw), directed=True, connection='strong')
    del C_raw
    largest   = pd.Series(labels).value_counts().idxmax()
    keep      = {all_stations[i] for i, l in enumerate(labels) if l == largest}

    all_stations   = sorted(keep)
    n              = len(all_stations)
    station_to_idx = {s: i for i, s in enumerate(all_stations)}
    idx_to_station = {i: s for s, i in station_to_idx.items()}

    print(f"      Stations after SCC:  {n} (dropped {n_raw - len(keep)})")
    return station_to_idx, idx_to_station, reference_date, station_meta


# ── pass 2: partition by day into parquet ─────────────────────────────────────
def partition_by_day(files, station_to_idx, reference_date, decay, bin_minutes, temp_dir):
    print("\n[2/3] Pass 2 — partitioning trips by day of week...")
    os.makedirs(temp_dir, exist_ok=True)
    n      = len(station_to_idx)
    n_bins = 60 * 24 // bin_minutes

    # open 7 parquet writers
    schema  = pa.schema([
        ('start_idx', pa.int32()),
        ('end_idx',   pa.int32()),
        ('bin',       pa.int16()),
        ('weight',    pa.float32()),
    ])
    writers = [
        pq.ParquetWriter(os.path.join(temp_dir, f'day_{d}.parquet'), schema)
        for d in range(7)
    ]

    # also accumulate C_global, C_per_day, arrivals, departures
    C_global        = np.zeros((n, n), dtype='float64')
    C_per_day       = np.zeros((7, n, n), dtype='float64')
    arrivals        = np.zeros((7, n_bins, n), dtype='float64')
    departures      = np.zeros((7, n_bins, n), dtype='float64')
    duration_sum    = np.zeros((7, n_bins), dtype='float64')  # weighted sum of durations (hours)
    duration_weight = np.zeros((7, n_bins), dtype='float64')  # total weight for averaging

    for i, f in enumerate(files):
        df = read_file(f)
        if df.empty:
            continue

        df = df[df['start_station_id'].isin(station_to_idx)]
        df = df[df['end_station_id'].isin(station_to_idx)]
        if df.empty:
            continue

        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df = df.dropna(subset=['started_at'])

        df['days_ago']    = (reference_date - df['started_at']).dt.days.clip(lower=0)
        df['weight']      = (decay ** df['days_ago']).astype('float32')
        df['day_of_week'] = df['started_at'].dt.dayofweek.astype('int8')
        total_minutes     = df['started_at'].dt.hour * 60 + df['started_at'].dt.minute
        df['bin']         = (total_minutes // bin_minutes).astype('int16')
        # trip duration in hours
        if 'ended_at' in df.columns:
            df['ended_at']  = pd.to_datetime(df['ended_at'], errors='coerce')
            df['duration_h'] = ((df['ended_at'] - df['started_at']).dt.total_seconds() / 3600)
            df['duration_h'] = df['duration_h'].clip(lower=0, upper=24)  # exclude bad data
        else:
            df['duration_h'] = np.nan
        df['start_idx']   = df['start_station_id'].map(station_to_idx).astype('int32')
        df['end_idx']     = df['end_station_id'].map(station_to_idx).astype('int32')

        si = df['start_idx'].values
        ei = df['end_idx'].values
        dd = df['day_of_week'].values
        bb = df['bin'].values
        w  = df['weight'].values

        np.add.at(C_global, (si, ei), w)

        for day in range(7):
            mask = dd == day
            if not mask.any():
                continue

            np.add.at(C_per_day[day], (si[mask], ei[mask]), w[mask])

            for b in range(n_bins):
                bm = mask & (bb == b)
                if bm.any():
                    np.add.at(arrivals[day, b],   ei[bm], w[bm])
                    np.add.at(departures[day, b], si[bm], w[bm])
                    if 'duration_h' in df.columns:
                        valid = bm & df['duration_h'].notna()
                        if valid.any():
                            dur_vals = df.loc[valid, 'duration_h'].values
                            dur_w    = w[valid.values]
                            duration_sum[day, b]    += (dur_vals * dur_w).sum()
                            duration_weight[day, b] += dur_w.sum()

            # write to parquet
            day_df = df[mask][['start_idx', 'end_idx', 'bin', 'weight']].copy()
            table  = pa.Table.from_pandas(day_df, schema=schema, preserve_index=False)
            writers[day].write_table(table)

        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(files)} files processed")

    for w in writers:
        w.close()

    print(f"      Parquet files written to {temp_dir}")
    return C_global, C_per_day, arrivals, departures, duration_sum, duration_weight


# ── pass 3: compute pi bin by bin from parquet ────────────────────────────────
def compute_all_pi(temp_dir, C_global, C_per_day, n, alpha1, alpha2, bin_minutes):
    n_bins = 60 * 24 // bin_minutes
    print(f"\n[3/3] Pass 3 — computing {7 * n_bins} stationary distributions...")

    pi_by_day_bin = np.zeros((7, n_bins, n))

    for day in range(7):
        path = os.path.join(temp_dir, f'day_{day}.parquet')
        df   = pd.read_parquet(path)   # ~1-2GB, one day at a time

        si = df['start_idx'].values
        ei = df['end_idx'].values
        bb = df['bin'].values
        w  = df['weight'].values

        for b in range(n_bins):
            mask  = bb == b
            C_bin = np.zeros((n, n), dtype='float64')
            if mask.any():
                np.add.at(C_bin, (si[mask], ei[mask]), w[mask])

            pi_by_day_bin[day, b] = compute_stationary(
                C_bin, C_per_day[day], C_global, alpha1, alpha2
            )

        del df   # free day's data before loading next
        print(f"      Day {day+1}/7 done")

    pi_by_bin = pi_by_day_bin.mean(axis=0)
    pi_by_bin /= pi_by_bin.sum(axis=1, keepdims=True)
    return pi_by_day_bin, pi_by_bin


def compute_stationary(C, C_day, C_global, alpha1, alpha2):
    C_s      = C + alpha1 * C_day + alpha2 * C_global
    row_sums = C_s.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P        = C_s / row_sums
    vals, vecs = linalg.eig(P.T)
    idx      = np.argmin(np.abs(vals - 1))
    pi       = vecs[:, idx].real
    return np.abs(pi) / np.abs(pi).sum()


def compute_flow_ratios(arrivals, departures, min_each):
    return np.where(
        (arrivals >= min_each) & (departures >= min_each),
        arrivals / (departures + 1e-10),
        1.0
    )


def build_stations_json(idx_to_station, station_meta):
    idx_to_info = {}
    for idx, sid in idx_to_station.items():
        meta = station_meta.get(sid, {})
        lat  = meta.get('lat')
        lng  = meta.get('lng')
        idx_to_info[idx] = {
            'name': meta.get('name', sid),
            'lat':  None if lat is None or (isinstance(lat, float) and np.isnan(lat)) else float(lat),
            'lng':  None if lng is None or (isinstance(lng, float) and np.isnan(lng)) else float(lng),
        }
    return idx_to_info


# ── main ──────────────────────────────────────────────────────────────────────
def main(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR,
         temp_dir=DEFAULT_TEMP_DIR, decay=DEFAULT_DECAY,
         min_each=DEFAULT_MIN_EACH, alpha1=DEFAULT_ALPHA1,
         alpha2=DEFAULT_ALPHA2, bin_minutes=DEFAULT_BIN_MINUTES,
         keep_temp=False):

    os.makedirs(output_dir, exist_ok=True)
    t0     = time.time()
    n_bins = 60 * 24 // bin_minutes
    print(f"\n  Bin size: {bin_minutes} min ({n_bins} bins/day, {7*n_bins} total)")

    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"  Found {len(files)} files")

    station_to_idx, idx_to_station, reference_date, station_meta = build_station_index(files)
    n = len(station_to_idx)

    C_global, C_per_day, arrivals, departures, duration_sum, duration_weight = partition_by_day(
        files, station_to_idx, reference_date, decay, bin_minutes, temp_dir
    )

    pi_by_day_bin, pi_by_bin = compute_all_pi(
        temp_dir, C_global, C_per_day, n, alpha1, alpha2, bin_minutes
    )

    print("\n[4/4] Saving outputs...")
    flow_ratio = compute_flow_ratios(arrivals, departures, min_each)

    # Little's Law: N_in_transit = avg_duration_hours × lambda_system (trips/hour)
    # N_docked = FLEET_SIZE - N_in_transit, varies by day and bin
    FLEET_SIZE         = 35_000
    avg_duration       = np.where(duration_weight > 0,
                                  duration_sum / duration_weight, 0.2)  # default 12min
    lambda_system      = arrivals.sum(axis=2) / (bin_minutes / 60)      # total trips/hour
    n_in_transit       = avg_duration * lambda_system
    n_docked           = np.clip(FLEET_SIZE - n_in_transit, 1, FLEET_SIZE)

    idx_to_info = build_stations_json(idx_to_station, station_meta)

    np.save(os.path.join(output_dir, 'pi_by_day_bin.npy'),         pi_by_day_bin)
    np.save(os.path.join(output_dir, 'pi_by_bin.npy'),             pi_by_bin)
    np.save(os.path.join(output_dir, 'flow_ratio_by_day_bin.npy'), flow_ratio)
    np.save(os.path.join(output_dir, 'arrivals_by_day_bin.npy'),     arrivals)
    np.save(os.path.join(output_dir, 'departures_by_day_bin.npy'),   departures)
    np.save(os.path.join(output_dir, 'n_docked_by_day_bin.npy'),     n_docked)
    np.save(os.path.join(output_dir, 'bin_config.npy'),            np.array([bin_minutes, n_bins]))
    with open(os.path.join(output_dir, 'stations.json'), 'w') as f:
        json.dump(idx_to_info, f)

    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"  Temp files cleaned up")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  pi_by_day_bin:         {pi_by_day_bin.shape}")
    print(f"  pi_by_bin:             {pi_by_bin.shape}")
    print(f"  flow_ratio_by_day_bin: {flow_ratio.shape}")
    print(f"  arrivals_by_day_bin:   {arrivals.shape}")
    print(f"  departures_by_day_bin: {departures.shape}")
    print(f"  n_docked_by_day_bin:   {n_docked.shape}  (mean={n_docked.mean():.0f})")
    print(f"  bin_minutes:           {bin_minutes} ({n_bins} bins/day)")
    print(f"  stations:              {len(idx_to_info)}")
    print(f"  outputs -> {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discrete-time Markov Chain pipeline")
    parser.add_argument('--data-dir',    default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--temp-dir',    default=DEFAULT_TEMP_DIR)
    parser.add_argument('--decay',       type=float, default=DEFAULT_DECAY)
    parser.add_argument('--min-each',    type=int,   default=DEFAULT_MIN_EACH)
    parser.add_argument('--alpha1',      type=float, default=DEFAULT_ALPHA1)
    parser.add_argument('--alpha2',      type=float, default=DEFAULT_ALPHA2)
    parser.add_argument('--bin-minutes', type=int,   default=DEFAULT_BIN_MINUTES,
                        help='Bin size in minutes (60=hourly, 30=half-hourly, 15=quarter-hourly)')
    parser.add_argument('--keep-temp',   action='store_true',
                        help='Keep temp parquet files after completion')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.temp_dir, args.decay,
         args.min_each, args.alpha1, args.alpha2, args.bin_minutes, args.keep_temp)