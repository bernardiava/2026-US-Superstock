"""Agri-Trade Dispute & Policy Monitor — Streamlit dashboard.

Jalankan: streamlit run agri_trade_monitor/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agri_trade_monitor.config import AGREEMENTS, DATA_PROCESSED_DIR, OUTPUT_DIR  # noqa: E402
from agri_trade_monitor import dispute_data  # noqa: E402

st.set_page_config(page_title="Agri-Trade Dispute & Policy Monitor",
                   page_icon="🌾", layout="wide")


@st.cache_data
def load_indicators() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED_DIR / "indicators_long.csv")


@st.cache_data
def load_disputes() -> pd.DataFrame:
    return dispute_data.load()


st.title("🌾 Agri-Trade Dispute & Policy Monitor")
st.caption(
    "Data-driven monitoring of agricultural trade frictions and the international "
    "agreement architecture that governs them — built for an Agricultural Trade "
    "Assistant workflow (post reporting, NTE follow-up, problem-solving casework)."
)

ind = load_indicators()
disp = load_disputes()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview", "⚖️ Trade Problem Resolution", "🤝 International Agreements",
    "📊 Indicator Analytics", "✅ Data Quality",
])

# ---------------------------------------------------------------- Overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disputes tracked", len(disp))
    c2.metric("US-involved cases", int(disp.apply(
        lambda r: "USA" in r.complainant or r.respondent == "USA" or "US -" in r.case, axis=1).sum()))
    c3.metric("Countries monitored", ind.iso3.nunique())
    c4.metric("Indicators scraped", ind.indicator.nunique())

    st.subheader("Dispute filings over time")
    by_year = disp.groupby("year").size().reset_index(name="cases")
    chart = alt.Chart(by_year).mark_bar(color="#8B0000").encode(
        x=alt.X("year:O", title=None), y="cases:Q", tooltip=["year", "cases"])
    st.altair_chart(chart.interactive(), use_container_width=True)

# ------------------------------------------------- Trade Problem Resolution
with tab2:
    st.subheader("⚖️ Trade Problem Resolution casework")
    st.markdown(
        "Pipeline view of market-access barriers: **identify → engage (TIFA/SPS "
        "Committee) → escalate (consultations/panel) → resolve & verify**. "
        "Filter by product, respondent, or agreement cited."
    )
    f_prod = st.multiselect("Product", sorted(disp.product.unique()))
    f_resp = st.multiselect("Respondent", sorted(disp.respondent.unique()))
    view = disp.copy()
    if f_prod:
        view = view[view.product.isin(f_prod)]
    if f_resp:
        view = view[view.respondent.isin(f_resp)]

    def stage(status: str) -> str:
        s = status.lower()
        if "resolved" in s or "repealed" in s or "settled" in s or "dismissed" in s or "terminated" in s:
            return "4. Resolved"
        if "compliance" in s or "implementation" in s or "surveillance" in s or "adopted" in s:
            return "3. Adjudicated – verifying implementation"
        if "panel" in s or "ruled" in s or "appeal" in s:
            return "3. Panel/Appellate phase"
        if "consultation" in s or "bilateral" in s or "talks" in s or "political" in s:
            return "2. Engagement / consultations"
        return "1. Monitoring barrier"

    view["stage"] = view.status.map(stage)
    st.altair_chart(
        alt.Chart(view).mark_point(size=90, filled=True, color="#C1440E").encode(
            x=alt.X("stage:N", title="Resolution stage", sort=sorted(view.stage.unique())),
            y=alt.Y("product:N", title="Product"),
            tooltip=["case", "respondent", "status", "year"]),
        use_container_width=True)
    st.dataframe(view[["case", "respondent", "product", "agreements", "status", "year"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Export filtered casework CSV",
                       view.to_csv(index=False).encode(), "casework_export.csv", "text/csv")

# --------------------------------------- International Agreements on Agri Policy
with tab3:
    st.subheader("🤝 International Agreements on Agricultural Policy")
    st.markdown(
        "The rules architecture behind every casework decision: WTO Agreement on "
        "Agriculture pillars, SPS/TFA disciplines, and preferential agreements "
        "(USMCA, RCEP, and pending Indonesia–EU CEPA) with their live friction points."
    )
    agr_df = pd.DataFrame(AGREEMENTS)
    for _, row in agr_df.iterrows():
        with st.expander(f"{row['name']}  —  effective {row['entered']}"):
            st.markdown(f"**Type:** {row['type']} · **Members covered:** {row['members_covered']}")
            st.markdown(f"**Pillars:** {'; '.join(row['pillars'])}")
            st.markdown(f"**Key mechanism:** {row['key_mechanism']}")
            st.markdown(f"**Relevance to dispute resolution:** {row['dispute_link']}")

    st.markdown("#### Agreements most cited in disputes")
    cite = pd.read_csv(OUTPUT_DIR / "agreement_citations.csv")
    st.bar_chart(cite.set_index(cite.columns[0]).iloc[:, 0].head(8))

# ------------------------------------------------------------ Indicator Analytics
with tab4:
    st.subheader("📊 World Bank indicator analytics")
    sel_ind = st.selectbox("Indicator", sorted(ind.indicator.unique()))
    sub = ind[ind.indicator == sel_ind]
    countries = st.multiselect("Countries", sorted(sub.country.unique()),
                               default=list(sorted(sub.country.unique()))[:6])
    sub = sub[sub.country.isin(countries)]
    line = alt.Chart(sub).mark_line().encode(
        x="year:O", y=alt.Y("value:Q", title=sel_ind),
        color="country:N", tooltip=["country", "year", "value"]).interactive()
    st.altair_chart(line, use_container_width=True)

    st.markdown("#### Correlation matrix (scraped indicators)")
    try:
        cm = pd.read_csv(OUTPUT_DIR / "correlation_matrix.csv", index_col=0)
        st.dataframe(cm.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                     use_container_width=True)
    except FileNotFoundError:
        st.info("Run analysis first.")

# ------------------------------------------------------------------ Data Quality
with tab5:
    st.subheader("✅ Data quality & provenance")
    try:
        q = pd.read_csv(OUTPUT_DIR / "data_quality_summary.csv")
        st.dataframe(q, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning("Run `python -m agri_trade_monitor.scraper` first.")
    st.markdown(
        "**Sources:** World Bank Open Data API (indicator series, 2010–2024); "
        "WTO Dispute Settlement database & USTR National Trade Estimate report "
        "(dispute corpus, simplified for portfolio use). Raw pulls are versioned "
        "under `agri_trade_monitor/data/raw/` for reproducibility."
    )
