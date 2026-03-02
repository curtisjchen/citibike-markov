"""
pipeline.py — Full Citibike pipeline orchestrator

Usage:
    uv run src/pipeline.py                  # run discrete pipeline only
    uv run src/pipeline.py --download       # download data first, then run
    uv run src/pipeline.py --download-only  # just download, skip computation
"""

import argparse
import os
import subprocess
import sys

SRC_DIR = os.path.dirname(__file__)


def run(script, extra_args=[]):
    path = os.path.join(SRC_DIR, script)
    print(f"\n{'='*50}")
    print(f"  {script}")
    print(f"{'='*50}")
    subprocess.run([sys.executable, path] + extra_args, check=True)


def main():
    parser = argparse.ArgumentParser(description="Full Citibike pipeline")
    parser.add_argument('--download',       action='store_true', help='Download data before computing')
    parser.add_argument('--download-only',  action='store_true', help='Only download, skip computation')
    parser.add_argument('--start',          default='202001',    help='Download start month yyyymm')
    parser.add_argument('--end',            default='202601',    help='Download end month yyyymm')
    parser.add_argument('--data-dir',       default=os.path.join(SRC_DIR, '..', 'data', 'raw'))
    parser.add_argument('--output-dir',     default=os.path.join(SRC_DIR, '..', 'outputs'))
    parser.add_argument('--decay',          default='0.9995')
    parser.add_argument('--min-each',       default='5')
    parser.add_argument('--alpha1',         default='0.1')
    parser.add_argument('--alpha2',         default='0.01')
    parser.add_argument('--bin-minutes',    default='60',
                        help='Bin size in minutes (60=hourly, 30=half-hourly, 15=quarter-hourly)')
    parser.add_argument('--temp-dir',       default=os.path.join(SRC_DIR, '..', 'temp'))
    parser.add_argument('--keep-temp',      action='store_true')
    args = parser.parse_args()

    if args.download or args.download_only:
        run('download_data.py', [
            '--start',    args.start,
            '--end',      args.end,
            '--data-dir', args.data_dir,
        ])

    if args.download_only:
        print("\nDownload complete.")
        return

    run('discrete_pipeline.py', [
        '--data-dir',   args.data_dir,
        '--output-dir', args.output_dir,
        '--decay',      args.decay,
        '--min-each',   args.min_each,
        '--alpha1',     args.alpha1,
        '--alpha2',     args.alpha2,
        '--bin-minutes', args.bin_minutes,
        '--temp-dir',    args.temp_dir,
        *([ '--keep-temp'] if args.keep_temp else []),
    ])

    # continuous_pipeline.py will go here once implemented
    # run('continuous_pipeline.py', ['--output-dir', args.output_dir])

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()