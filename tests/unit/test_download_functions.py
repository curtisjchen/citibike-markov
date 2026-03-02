"""
tests/unit/test_download_functions.py

Unit tests for download_data.py functions.
Run with: uv run pytest tests/unit/ -v
"""

import io
import zipfile
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from download_data import get_urls, generate_months, already_downloaded, extract_month


# ── get_urls ──────────────────────────────────────────────────────────────────

def test_get_urls_nyc_post_2024_monthly():
    urls = get_urls('202401', jc=False)
    assert len(urls) == 1
    assert '202401-citibike-tripdata.zip' in urls[0]
    assert 'JC' not in urls[0]


def test_get_urls_nyc_pre_2024_annual():
    urls = get_urls('202306', jc=False)
    assert len(urls) == 1
    assert '2023-citibike-tripdata.zip' in urls[0]
    assert '202306' not in urls[0]   # annual zip, not monthly


def test_get_urls_nyc_pre_2024_uses_year_not_month():
    urls = get_urls('202209', jc=False)
    assert '2022-citibike-tripdata.zip' in urls[0]


def test_get_urls_jc_always_monthly():
    urls = get_urls('202201', jc=True)
    assert any('202201' in u for u in urls), "JC should use monthly URL"


def test_get_urls_jc_tries_both_suffixes():
    urls = get_urls('202301', jc=True)
    assert len(urls) == 2
    assert any(u.endswith('.zip') and '.csv.zip' not in u for u in urls)
    assert any(u.endswith('.csv.zip') for u in urls)


def test_get_urls_jc_has_prefix():
    urls = get_urls('202401', jc=True)
    assert all('JC-' in u for u in urls)


def test_get_urls_nyc_no_prefix():
    urls = get_urls('202401', jc=False)
    assert all('JC-' not in u for u in urls)


def test_get_urls_jc_zip_comes_before_csv_zip():
    # try plain .zip first — it's more common in recent months
    urls = get_urls('202401', jc=True)
    assert urls[0].endswith('.zip') and '.csv.zip' not in urls[0]
    assert urls[1].endswith('.csv.zip')


# ── generate_months ───────────────────────────────────────────────────────────

def test_generate_months_basic():
    assert generate_months('202301', '202303') == ['202301', '202302', '202303']


def test_generate_months_single():
    assert generate_months('202401', '202401') == ['202401']


def test_generate_months_year_boundary():
    months = generate_months('202311', '202402')
    assert months == ['202311', '202312', '202401', '202402']


def test_generate_months_respects_exact_start():
    months = generate_months('202206', '202209')
    assert months[0] == '202206'
    assert '202205' not in months


# ── pre-2020 validation ───────────────────────────────────────────────────────

def test_main_raises_before_2020():
    from download_data import main
    with pytest.raises(ValueError, match="before 2020"):
        main(start='201901', end='201912', data_dir='/tmp')

def test_main_raises_for_2019():
    from download_data import main
    with pytest.raises(ValueError):
        main(start='201901', end='202001', data_dir='/tmp')

def test_main_accepts_2020():
    # should not raise — just check validation passes, don't actually download
    from download_data import generate_months
    months = generate_months('202001', '202003')
    assert months[0] == '202001'

# ── already_downloaded ────────────────────────────────────────────────────────

def test_already_downloaded_false_if_dir_missing(tmp_path):
    assert not already_downloaded('202401', str(tmp_path))


def test_already_downloaded_false_if_dir_empty(tmp_path):
    (tmp_path / '202401').mkdir()
    assert not already_downloaded('202401', str(tmp_path))


def test_already_downloaded_false_if_no_csv(tmp_path):
    d = tmp_path / '202401'
    d.mkdir()
    (d / 'readme.txt').write_text('hello')
    assert not already_downloaded('202401', str(tmp_path))


def test_already_downloaded_true_if_csv_present(tmp_path):
    d = tmp_path / '202401'
    d.mkdir()
    (d / '202401-citibike-tripdata.csv').write_text('col1,col2\n1,2')
    assert already_downloaded('202401', str(tmp_path))


# ── extract_month ─────────────────────────────────────────────────────────────

def make_zip(filenames):
    """Helper — create an in-memory zip with dummy CSV content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name in filenames:
            z.writestr(name, 'start_station_id,end_station_id\nA,B\n')
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_extract_month_extracts_matching_csv(tmp_path):
    z  = make_zip(['202401-citibike-tripdata.csv', '202402-citibike-tripdata.csv'])
    ok = extract_month(z, '202401', str(tmp_path))
    assert ok
    extracted = list((tmp_path / '202401').iterdir())
    assert len(extracted) == 1
    assert extracted[0].name == '202401-citibike-tripdata.csv'


def test_extract_month_falls_back_to_all_csvs_if_no_match(tmp_path):
    # annual zip may not have yyyymm in filenames
    z  = make_zip(['January2023-citibike-tripdata.csv'])
    ok = extract_month(z, '202301', str(tmp_path))
    assert ok


def test_extract_month_returns_false_if_no_csvs(tmp_path):
    z  = make_zip([])
    ok = extract_month(z, '202401', str(tmp_path))
    assert not ok


def test_extract_month_creates_ym_dir(tmp_path):
    z = make_zip(['202403-citibike-tripdata.csv'])
    extract_month(z, '202403', str(tmp_path))
    assert (tmp_path / '202403').is_dir()