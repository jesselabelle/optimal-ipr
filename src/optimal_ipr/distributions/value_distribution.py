"""Construct the patent value distribution from the KPSS 2023 dataset."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
ZIP_FILENAME = "KPSS_2023.csv.zip"
CSV_FILENAME = "KPSS_2023.csv"
DEFAULT_YEAR_RANGE: Tuple[int, int] = (2010, 2019)
GAUSS_HERMITE_NODES = 101

def _coerce_nominal(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype(str)
        .str.replace(r"[,\s$]", "", regex=True)
        .str.replace("−", "-", regex=False)
        .str.replace("\u2212", "-", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_year(issue_date: pd.Series) -> pd.Series:
    raw = issue_date.astype(str).str.strip()
    parsed = pd.to_datetime(raw, errors="coerce", infer_datetime_format=True)
    year = parsed.dt.year
    if year.notna().sum() == 0:
        extracted = raw.str.extract(r"/(\d{4})\\b", expand=False)
        year = pd.to_numeric(extracted, errors="coerce")
    return year


def _load_patent_values(
    zip_path: Path, csv_name: str, year_range: Tuple[int, int]
) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_name) as handle:
            df = pd.read_csv(handle, low_memory=False)
    df["xi_nominal"] = _coerce_nominal(df["xi_nominal"])
    df["year"] = _extract_year(df["issue_date"])

    yr0, yr1 = year_range
    mask = df["year"].between(yr0, yr1)
    sample = df.loc[mask, "xi_nominal"].dropna().to_numpy()
    if sample.size == 0:
        yr_min = df["year"].min()
        yr_max = df["year"].max()
        raise ValueError(
            "No patent value observations within the requested range"
            f" [{yr0}, {yr1}]; available span=({yr_min}, {yr_max})"
        )
    return sample


def value_distribution(
    *,
    data_dir: Path | None = None,
    year_range: Tuple[int, int] = DEFAULT_YEAR_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (v_grid, v_weights) using patent values from the KPSS dataset."""

    base_dir = data_dir or DATA_DIR
    zip_path = base_dir / ZIP_FILENAME
    if not zip_path.exists():
        raise FileNotFoundError(
             f"Patent value archive not found at {zip_path}. Place {ZIP_FILENAME} in the data directory."
        )

    sample = _load_patent_values(zip_path, CSV_FILENAME, year_range)
    log_sample = np.log(sample)
    mu, sigma_sample = log_sample.mean(), log_sample.std(ddof=0)

    nodes, weights = hermgauss(GAUSS_HERMITE_NODES)
    v_weights = weights / np.sqrt(np.pi)
    v_grid = np.exp(mu + np.sqrt(2.0) * sigma_sample * nodes)

    order = np.argsort(v_grid)
    v_grid = v_grid[order]
    v_weights = v_weights[order]

    cutoff_mask = v_grid > 0.1
    if not np.any(cutoff_mask):
        raise ValueError("Lognormal quadrature grid does not exceed 0.1 after fitting.")
    first_idx = np.argmax(cutoff_mask)
    v_grid = v_grid[first_idx:]
    v_weights = v_weights[first_idx:]
    v_weights = v_weights / v_weights.sum()

    return v_grid, v_weights