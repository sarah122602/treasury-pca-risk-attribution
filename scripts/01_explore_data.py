"""Download and explore official U.S. Treasury par yields.

This is Step 1 only. It intentionally does not implement PCA, hedging, or
regression, and it never fills missing observations.
"""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
START_YEAR = 2015
TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


def treasury_url(year: int) -> str:
    """Return Treasury's official annual daily par-yield CSV URL."""
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all"
        "?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}"
    )


def _normalise_column(name: object) -> str:
    """Normalize Treasury labels such as ``2 Yr`` and ``20 YR`` to ``2Y``."""
    text = re.sub(r"[^0-9a-zA-Z.]", " ", str(name)).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:y|yr|yrs|year|years)", text, re.IGNORECASE)
    if match:
        return f"{float(match.group(1)):g}Y"
    return text.replace(" ", "").upper()


def download_year(year: int, inspect_response: bool = False) -> pd.DataFrame:
    """Download one year, validate it is CSV, and return the raw parsed table."""
    response = requests.get(treasury_url(year), timeout=30)
    response.raise_for_status()
    body_start = response.content.lstrip()[:200].lower()
    content_type = response.headers.get("content-type", "")
    if body_start.startswith(b"<!doctype html") or b"<html" in body_start:
        preview = response.text[:500].replace("\n", " ")
        raise ValueError(
            f"Treasury returned HTML instead of CSV for {year} "
            f"(content-type={content_type!r}): {preview}"
        )

    try:
        parsed = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        preview = response.text[:500].replace("\n", " ")
        raise ValueError(
            f"Treasury response for {year} is not a valid CSV "
            f"(content-type={content_type!r}): {preview}"
        ) from exc

    if inspect_response:
        print(f"\nTreasury response inspection for {year}")
        print(f"Content-Type: {content_type or '<not provided>'}")
        print(f"Actual columns: {list(parsed.columns)}")
        print("Sample rows:")
        print(parsed.head(3).to_string(index=False))
    return parsed


def clean_yields(raw: pd.DataFrame) -> pd.DataFrame:
    """Select and clean date plus requested tenors without imputing values."""
    date_column = next(
        (column for column in raw.columns if str(column).strip().lower() == "date"),
        None,
    )
    if date_column is None:
        raise ValueError("Treasury response does not contain a Date column")

    normalized = {_normalise_column(column): column for column in raw.columns}
    missing_columns = [tenor for tenor in TENORS if tenor.upper() not in normalized]
    if missing_columns:
        raise ValueError(
            "Treasury response is missing requested tenors: "
            + ", ".join(missing_columns)
        )

    selected = raw[[date_column] + [normalized[tenor.upper()] for tenor in TENORS]].copy()
    selected.columns = ["Date"] + TENORS
    selected["Date"] = pd.to_datetime(selected["Date"], errors="coerce")
    for tenor in TENORS:
        selected[tenor] = pd.to_numeric(selected[tenor], errors="coerce")

    selected = selected.dropna(subset=["Date"]).sort_values("Date")
    selected = selected.drop_duplicates(subset=["Date"], keep="last")
    return selected.reset_index(drop=True)


def save_exploration_outputs(yields: pd.DataFrame) -> None:
    """Save cleaned data, changes, and the two Step 1 figures."""
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    yields.to_csv(DATA_DIR / "treasury_par_yields.csv", index=False, date_format="%Y-%m-%d")
    changes = yields.set_index("Date")[TENORS].diff() * 100.0
    changes.reset_index().to_csv(
        DATA_DIR / "treasury_daily_changes_bp.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7))
    for tenor in TENORS:
        ax.plot(yields["Date"], yields[tenor], label=tenor, linewidth=1.0)
    ax.set_title("U.S. Treasury Par Yields")
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "treasury_yield_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(changes.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation of Daily Treasury Yield Changes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "treasury_yield_correlation.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Download through the latest available calendar year and report results."""
    frames: list[pd.DataFrame] = []
    latest_year = date.today().year
    for year in range(START_YEAR, latest_year + 1):
        print(f"Downloading Treasury data for {year} ...")
        frames.append(download_year(year, inspect_response=not frames))

    yields = clean_yields(pd.concat(frames, ignore_index=True))
    save_exploration_outputs(yields)

    changes = yields.set_index("Date")[TENORS].diff() * 100.0
    print(f"\nRows: {len(yields):,}")
    print(f"Date range: {yields['Date'].min().date()} to {yields['Date'].max().date()}")
    print("\nMissing values in cleaned yields:")
    print(yields[TENORS].isna().sum().to_string())
    print("\nYield summary statistics (percent):")
    print(yields[TENORS].describe().round(4).to_string())
    print("\nDaily-change summary statistics (basis points):")
    print(changes.describe().round(4).to_string())
    print("\nSaved data/treasury_par_yields.csv")
    print("Saved data/treasury_daily_changes_bp.csv")
    print("Saved figures/treasury_yield_history.png")
    print("Saved figures/treasury_yield_correlation.png")


if __name__ == "__main__":
    main()

