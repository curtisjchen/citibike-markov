"""
download_data.py — Download Citibike trip data from S3

Usage:
    uv run src/download_data.py
    uv run src/download_data.py --start 202401 --end 202601
    uv run src/download_data.py --start 202401 --end 202601 --data-dir data/raw
"""

import argparse
import io
import os
import zipfile

import requests

BASE_URL         = 'https://s3.amazonaws.com/tripdata'
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DEFAULT_START    = '202001'
DEFAULT_END      = '202601'


# pre-2024 data is annual zip containing all monthly CSVs for that year
# 2024+ data is one zip per month
ANNUAL_URL_CUTOFF = 2024


def generate_months(start_ym, end_ym):
    """Generate yyyymm strings for all months in range (inclusive)."""
    months = []
    y, m = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]), int(end_ym[4:])
    while (y, m) <= (ey, em):
        months.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def already_downloaded(label, save_dir):
    ym_dir = os.path.join(save_dir, label)
    return os.path.isdir(ym_dir) and any(f.endswith('.csv') for f in os.listdir(ym_dir))


def get_urls(ym, jc=False):
    """Return list of candidate URLs to try in order.
    NYC pre-2024: annual zip. NYC 2024+: monthly zip.
    JC: always monthly, but suffix varies between .zip and .csv.zip.
    """
    year   = int(ym[:4])
    prefix = "JC-" if jc else ""
    if jc:
        # try both suffixes — older JC files use .csv.zip
        return [
            f"{BASE_URL}/{prefix}{ym}-citibike-tripdata.zip",
            f"{BASE_URL}/{prefix}{ym}-citibike-tripdata.csv.zip",
        ]
    elif year >= ANNUAL_URL_CUTOFF:
        return [f"{BASE_URL}/{ym}-citibike-tripdata.zip"]
    else:
        return [f"{BASE_URL}/{year}-citibike-tripdata.zip"]


def fetch_zip(urls, label):
    """Try each URL in order, return BytesIO of first success or None."""
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        print(f"[{label}] Trying {url}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            if response.status_code == 404:
                print(f"[{label}] 404 — trying next URL...")
                continue
            if response.status_code != 200:
                print(f"[{label}] HTTP {response.status_code} — skipping")
                return None

            total      = int(response.headers.get('content-length', 0))
            data       = bytearray()
            downloaded = 0

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                data.extend(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r[{label}] {downloaded/total*100:.1f}% ({downloaded/1e6:.1f} MB)",
                          end='', flush=True)
            print()
            return io.BytesIO(data)

        except Exception as e:
            print(f"[{label}] Error: {e}")
            return None

    print(f"[{label}] Not found — month may not exist yet, skipping")
    return None


def extract_flat(z, member, dest_dir):
    """Extract a single zip member flat into dest_dir (no subdirectory nesting)."""
    filename = os.path.basename(member)
    if not filename:
        return
    dest = os.path.join(dest_dir, filename)
    with z.open(member) as src, open(dest, 'wb') as dst:
        dst.write(src.read())
    return filename


def extract_month(z, ym, save_dir, dest_dir=None):
    """Extract CSVs matching yyyymm from a ZipFile into dest_dir (defaults to save_dir/yyyymm/).
    
    Handles two structures:
    - Flat zip: zip contains CSVs directly
    - Nested zip: zip contains inner .zip files which contain CSVs (pre-2024 NYC annual format)
    """
    ym_dir = dest_dir if dest_dir is not None else os.path.join(save_dir, ym)
    os.makedirs(ym_dir, exist_ok=True)
    names  = z.namelist()

    # check if this zip contains inner zips matching our month
    inner_zips = [f for f in names if f.endswith('.zip') and ym in f
                   and '__MACOSX' not in f and not os.path.basename(f).startswith('._')]
    if inner_zips:
        # nested zip — extract inner zip for this month then get CSVs from it
        for inner_zip_path in inner_zips:
            inner_bytes = z.read(inner_zip_path)
            try:
                inner_z   = zipfile.ZipFile(io.BytesIO(inner_bytes))
                csv_files = [f for f in inner_z.namelist() if f.endswith('.csv') and '__MACOSX' not in f]
                for csv_file in csv_files:
                    fname = extract_flat(inner_z, csv_file, ym_dir)
                    print(f"[{ym}] Extracted {fname}")
            except zipfile.BadZipFile:
                # not actually a zip — treat as CSV directly
                fname = extract_flat(z, inner_zip_path, ym_dir)
                print(f"[{ym}] Extracted {fname} (not a zip)")
        return len(inner_zips) > 0

    # flat zip — CSVs are directly inside
    csv_files = [f for f in names if f.endswith('.csv') and ym in f and '__MACOSX' not in f]
    if not csv_files:
        csv_files = [f for f in names if f.endswith('.csv') and '__MACOSX' not in f]
    for csv_file in csv_files:
        fname = extract_flat(z, csv_file, ym_dir)
        print(f"[{ym}] Extracted {fname}")
    return len(csv_files) > 0


def download_and_extract(ym, save_dir, annual_cache={}, jc=False):
    """Download and extract a single month. For pre-2024, reuse cached annual zip."""
    year       = int(ym[:4])
    cache_key  = f"{'jc' if jc else 'nyc'}-{year}"
    label      = f"{'JC-' if jc else ''}{ym}"

    if jc or year >= ANNUAL_URL_CUTOFF:
        # JC is always monthly; NYC 2024+ is monthly
        buf = fetch_zip(get_urls(ym, jc), label)
        
        if buf is None:
            return False
        z = zipfile.ZipFile(buf)
        return extract_month(z, ym, save_dir, dest_dir=save_dir)

    else:
        # NYC pre-2024 — annual zip cached per year
        if cache_key not in annual_cache:
            buf = fetch_zip(get_urls(ym, jc), str(year))
            if buf is None:
                return False
            annual_cache[cache_key] = zipfile.ZipFile(buf)
            print(f"[{year}] Annual zip cached — extracting months individually")
        z = annual_cache[cache_key]
        return extract_month(z, ym, save_dir, dest_dir=save_dir)


def main(start=DEFAULT_START, end=DEFAULT_END, data_dir=DEFAULT_DATA_DIR):
    if int(start[:4]) < 2020:
        raise ValueError(
            f"Start date {start} is before 2020. Data before 2020 uses a different "
            f"schema and station ID format — please use --start 202001 or later."
        )
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {os.path.abspath(data_dir)}\n")

    months            = generate_months(start, end)
    succeeded, failed = [], []
    skipped           = []
    annual_cache      = {}   # cache key -> ZipFile, shared across months in the same year

    for jc in [False, True]:
        print(f"\n--- {"Jersey City" if jc else "New York City"} ---")
        for ym in months:
            label  = f"{'JC-' if jc else ''}{ym}"
            ym_dir = os.path.join(data_dir, label)
            if already_downloaded(label, data_dir):
                print(f"[{label}] already downloaded — skipping")
                skipped.append(label)
                continue
            ok = download_and_extract(ym, os.path.join(data_dir, label), annual_cache, jc)
            (succeeded if ok else failed).append(label)

    print(f"\n{'='*40}")
    print(f"Downloaded: {len(succeeded)} months")
    print(f"Skipped:    {len(skipped)} months (already present)")
    if failed:
        print(f"Not found / failed: {len(failed)} months -> {failed}")
        print(f"  (404s are expected for future months or missing JC data)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Citibike trip data")
    parser.add_argument('--start',    default=DEFAULT_START)
    parser.add_argument('--end',      default=DEFAULT_END)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    main(args.start, args.end, args.data_dir)