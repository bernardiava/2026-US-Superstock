"""Scraping / penarikan data indikator pertanian & perdagangan dari World Bank API.

Menyimpan hasil mentah ke data/raw/ dan versi bersih (long & wide CSV) ke
data/processed/. Jalankan:  python -m agri_trade_monitor.scraper
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agri_trade_monitor.config import (  # noqa: E402
    COUNTRIES, DATE_RANGE, INDICATORS, OUTPUT_DIR, REQUEST_TIMEOUT,
    DATA_RAW_DIR, DATA_PROCESSED_DIR, WB_BASE_URL,
)


def fetch_indicator(code: str, label: str) -> pd.DataFrame:
    """Ambil satu indikator untuk semua negara yang dipantau.

    API World Bank mengembalikan ISO2 pada field country.id, sehingga kami
    memetakan balik ke ISO3 via config.COUNTRIES.
    """
    iso_list = ";".join(COUNTRIES.keys())
    url = (
        f"{WB_BASE_URL}/country/{iso_list}"
        f"/indicator/{code}?format=json&per_page=2000&date={DATE_RANGE}"
    )
    # Build ISO2 -> ISO3 map from the first meta call? Simpler: hardcode common ones.
    iso2_to_iso3 = {
        "ID": "IDN", "MY": "MYS", "TH": "THA", "VN": "VNM", "PH": "PHL", "SG": "SGP",
        "CN": "CHN", "JP": "JPN", "KR": "KOR", "IN": "IND", "PK": "PAK", "BD": "BGD",
        "AU": "AUS", "NZ": "NZL", "US": "USA", "CA": "CAN", "MX": "MEX", "BR": "BRA",
        "AR": "ARG", "CO": "COL", "CR": "CRI", "EU": "EU", "GB": "GBR",
    }
    rows = []
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            if not isinstance(body, list) or len(body) < 2:
                print(f"  unexpected response for {code}: {body}")
                break
            meta, data = body[0], body[1]
            for rec in data or []:
                if rec["value"] is None:
                    continue
                raw_id = rec["country"]["id"]
                iso3 = iso2_to_iso3.get(raw_id, raw_id)
                rows.append({
                    "iso3": iso3,
                    "country": COUNTRIES.get(iso3, rec["country"]["value"]),
                    "year": int(rec["date"]),
                    "indicator_code": code,
                    "indicator": label,
                    "value": float(rec["value"]),
                })
            break
        except requests.RequestException as exc:
            print(f"  retry {attempt+1} for {code}: {exc}")
            time.sleep(2 * (attempt + 1))
    print(f"[OK] {label}: {len(rows)} observasi")
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [fetch_indicator(code, label) for code, label in INDICATORS.items()]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    df = df.sort_values(["iso3", "indicator_code", "year"]).reset_index(drop=True)

    # Simpan mentah + bersih
    df.to_csv(DATA_RAW_DIR / "worldbank_indicators_raw.csv", index=False)
    df_long = df.copy()
    df_wide = df.pivot_table(
        index=["iso3", "country", "year"],
        columns="indicator_code", values="value", aggfunc="mean",
    ).reset_index()
    df_long.to_csv(DATA_PROCESSED_DIR / "indicators_long.csv", index=False)
    df_wide.to_csv(DATA_PROCESSED_DIR / "indicators_wide.csv", index=False)

    # Ringkasan kualitas data untuk dashboard
    quality = (
        df.groupby("indicator")
        .agg(observations=("value", "size"), countries=("iso3", "nunique"),
             first_year=("year", "min"), last_year=("year", "max"))
        .reset_index()
    )
    quality.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    print(json.dumps({"total_rows": len(df),
                      "indicators": df['indicator'].nunique(),
                      "countries": df['iso3'].nunique()}))
    return df_long, df_wide


if __name__ == "__main__":
    run()
