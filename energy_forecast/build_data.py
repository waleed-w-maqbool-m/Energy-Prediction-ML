"""CLI: rebuild ``data/pjme_daily.csv`` and ``data/weather_daily.csv`` from source.

    python -m energy_forecast.build_data                  # downloads PJME via kagglehub
    python -m energy_forecast.build_data --hourly-csv PJME_hourly.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config
from .data import build_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--hourly-csv", type=Path, help="local PJME_hourly.csv (skips the Kaggle download)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    hourly_csv = args.hourly_csv
    if hourly_csv is None:
        import kagglehub  # only needed for rebuilds; kept out of runtime requirements

        hourly_csv = Path(kagglehub.dataset_download(config.KAGGLE_DATASET)) / config.KAGGLE_FILE

    df = build_data(hourly_csv)
    print(f"Built {len(df):,} days: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
