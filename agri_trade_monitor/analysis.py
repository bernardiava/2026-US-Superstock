"""Analisis: korelasi indikator pertanian/perdagangan dengan intensitas sengketa.

Output:
- output/dispute_analysis.csv  (pivot per negara & agreement)
- output/correlation_matrix.csv
- output/charts/*.png          (4 visualisasi siap pakai untuk portofolio)

Jalankan: python -m agri_trade_monitor.analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agri_trade_monitor.config import DATA_PROCESSED_DIR, OUTPUT_DIR  # noqa: E402
from agri_trade_monitor import dispute_data  # noqa: E402

CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_indicators() -> pd.DataFrame:
    p = DATA_PROCESSED_DIR / "indicators_long.csv"
    if not p.exists():
        raise FileNotFoundError("Run scraper first: python -m agri_trade_monitor.scraper")
    return pd.read_csv(p)


def us_involved(row: pd.Series) -> bool:
    return "USA" in row["complainant"] or row["respondent"] == "USA" \
        or "US " in row["case"] or "(US)" in row["case"]


def analyze() -> dict[str, pd.DataFrame]:
    ind = load_indicators()
    disp = dispute_data.load()
    disp["us_involved"] = disp.apply(us_involved, axis=1)

    # 1. Sengketa per tahun (semua vs yang melibatkan AS)
    by_year = (disp.groupby("year")
               .agg(disputes=("case", "count"), us_disputes=("us_involved", "sum"))
               .reset_index())

    # 2. Agreement yang dikutip paling sering
    agr_counts = (disp.assign(a=disp["agreements"].str.split(","))
                  .explode("a").assign(a=lambda d: d["a"].str.strip())
                  .groupby("a")["case"].count().sort_values(ascending=False))

    # 3. Respondent hotspot (negara target inisiatif perdagangan AS)
    resp_counts = disp["respondent"].value_counts().head(10)

    # 4. Korelasi openness perdagangan (Trade % GDP) dengan jumlah sengketa
    #    per negara (agregat sepanjang waktu) — proxy: makin terintegrasi,
    #    makin banyak interaksi -> potensi sengketa.
    trade_open = (ind[ind["indicator_code"] == "NE.TRD.GNFS.ZS"]
                  .groupby("iso3")["value"].mean().rename("avg_trade_pct_gdp"))
    resp_map = {"China": "CHN", "EU": "EU", "Indonesia": "IDN", "India": "IND",
                "Korea": "KOR", "Japan": "JPN", "Mexico": "MEX", "Canada": "CAN",
                "Turkey": None, "Russia": None, "Philippines": "PHL",
                "Thailand": "THA", "Vietnam": None, "Australia": "AUS",
                "Egypt": None, "Colombia": "COL", "Brazil": "BRA", "USA": "USA"}
    disp["resp_iso"] = disp["respondent"].map(lambda x: resp_map.get(x.split("/")[0].strip(), None))
    disp_per_country = disp.dropna(subset=["resp_iso"]).groupby("resp_iso").size()
    joined = pd.concat([trade_open, disp_per_country.rename("disputes")], axis=1).dropna()
    corr = joined.corr(method="pearson").loc["avg_trade_pct_gdp", "disputes"]

    # 5. Matriks korelasi antar-indikator (wide)
    wide = pd.read_csv(DATA_PROCESSED_DIR / "indicators_wide.csv")
    corr_matrix = wide.select_dtypes("number").corr().round(3)

    # Simpan
    by_year.to_csv(OUTPUT_DIR / "disputes_by_year.csv", index=False)
    agr_counts.to_csv(OUTPUT_DIR / "agreement_citations.csv")
    joined.to_csv(OUTPUT_DIR / "openness_vs_disputes.csv")
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    disp.to_csv(OUTPUT_DIR / "disputes_enriched.csv", index=False)

    return {"by_year": by_year, "agreements": agr_counts,
            "respondents": resp_counts, "joined": joined,
            "corr": corr, "corr_matrix": corr_matrix, "disputes": disp}


def make_charts(res: dict) -> list[Path]:
    paths = []

    # Chart 1: timeline sengketa
    fig, ax = plt.subplots(figsize=(10, 5))
    by = res["by_year"]
    ax.bar(by["year"], by["disputes"], color="#B0C4DE", label="All agri-related disputes")
    ax.bar(by["year"], by["us_disputes"], color="#8B0000", label="US-involved")
    ax.set_title("Agricultural Trade Disputes Filed per Year (WTO/PTA track)")
    ax.set_ylabel("Cases"); ax.legend(); fig.tight_layout()
    p = CHART_DIR / "01_disputes_timeline.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Chart 2: agreement citations
    fig, ax = plt.subplots(figsize=(9, 5))
    a = res["agreements"].head(10)
    ax.barh(a.index[::-1], a.values[::-1], color="#2E5E4E")
    ax.set_title("Agreements Cited in Agri Trade Disputes")
    ax.set_xlabel("Number of cases"); fig.tight_layout()
    p = CHART_DIR / "02_agreement_citations.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Chart 3: scatter openness vs disputes + trendline
    j = res["joined"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(j["avg_trade_pct_gdp"], j["disputes"], s=100, color="#C1440E")
    coef = np.polyfit(j["avg_trade_pct_gdp"], j["disputes"], 1)
    xs = np.linspace(j["avg_trade_pct_gdp"].min(), j["avg_trade_pct_gdp"].max(), 50)
    ax.plot(xs, np.polyval(coef, xs), "--", color="gray",
            label=f"Pearson r = {res['corr']:.2f}")
    for iso, r in j.iterrows():
        ax.annotate(iso, (r["avg_trade_pct_gdp"], r["disputes"]), fontsize=8)
    ax.set_xlabel("Average trade openness (Trade % of GDP, World Bank)")
    ax.set_ylabel("Agri disputes targeting country")
    ax.set_title("Trade Openness vs Agricultural Dispute Frequency")
    ax.legend(); fig.tight_layout()
    p = CHART_DIR / "03_openness_vs_disputes.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Chart 4: heatmap korelasi indikator
    cm = res["corr_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cm.index)), cm.index, fontsize=7)
    fig.colorbar(im); ax.set_title("Indicator Correlation Matrix (World Bank data)")
    fig.tight_layout()
    p = CHART_DIR / "04_indicator_correlation.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


if __name__ == "__main__":
    res = analyze()
    charts = make_charts(res)
    print(f"Openness-dispute correlation: r={res['corr']:.2f}")
    for c in charts:
        print("chart:", c)
