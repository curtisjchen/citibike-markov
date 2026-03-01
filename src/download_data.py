import requests
import zipfile
import io
import os
from datetime import datetime

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
BASE_URL  = 'https://s3.amazonaws.com/tripdata'

def generate_urls(start_ym, end_ym):
    """Generate (url, label) pairs for all months in range (inclusive)."""
    urls = []
    y, m = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]), int(end_ym[4:])

    while (y, m) <= (ey, em):
        ym = f"{y}{m:02d}"
        urls.append((f"{BASE_URL}/{ym}-citibike-tripdata.zip", ym))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return urls

def download_and_extract(url, label, save_dir):
    print(f"[{label}] Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        if response.status_code != 200:
            print(f"[{label}] ✗ HTTP {response.status_code} — skipping")
            return False

        total = int(response.headers.get('content-length', 0))
        data  = bytearray()
        downloaded = 0

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            data.extend(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r[{label}] {pct:.1f}% ({downloaded/1e6:.1f} MB)", end='', flush=True)

        print()

        z = zipfile.ZipFile(io.BytesIO(data))
        ym_dir = os.path.join(save_dir, label)
        os.makedirs(ym_dir, exist_ok=True)
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        for csv_file in csv_files:
            z.extract(csv_file, ym_dir)
            print(f"[{label}] ✓ Extracted {csv_file}")

        return True

    except Exception as e:
        print(f"[{label}] ✗ Error: {e}")
        return False

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving to: {os.path.abspath(SAVE_DIR)}\n")

    urls = generate_urls('202401', '202601')
    succeeded, failed = [], []

    for url, label in urls:
        ok = download_and_extract(url, label, SAVE_DIR)
        (succeeded if ok else failed).append(label)

    print(f"\n{'='*40}")
    print(f"✓ Downloaded: {len(succeeded)} months")
    if failed:
        print(f"✗ Failed:     {len(failed)} months → {failed}")

if __name__ == '__main__':
    main()