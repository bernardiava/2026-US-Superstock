"""Konfigurasi proyek Agri-Trade Dispute & Policy Monitor."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

WB_BASE_URL = "https://api.worldbank.org/v2"
REQUEST_TIMEOUT = 30
DATE_RANGE = "2010:2024"

# Negara yang dipantau: mitra dagang pertanian AS + pasar kunci Asia/Pasifik
COUNTRIES = {
    # ASEAN + mitra Asia Pasifik
    "IDN": "Indonesia", "MYS": "Malaysia", "THA": "Thailand",
    "VNM": "Vietnam", "PHL": "Philippines", "SGP": "Singapore",
    "CHN": "China", "JPN": "Japan", "KOR": "Korea, Rep.",
    "IND": "India", "PAK": "Pakistan", "BGD": "Bangladesh",
    "AUS": "Australia", "NZL": "New Zealand",
    # Mitra pertanian utama AS
    "USA": "United States", "CAN": "Canada", "MEX": "Mexico",
    "BRA": "Brazil", "ARG": "Argentina", "COL": "Colombia",
    "CRI": "Costa Rica", "EU": "European Union", "GBR": "United Kingdom",
}

# Indikator World Bank relevan dengan perdagangan & kebijakan pertanian
INDICATORS = {
    "AG.LND.AGRI.ZS": "Agricultural land (% of land area)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP) - openness indicator",
    "NV.AGR.TOTL.ZS": "Agriculture, forestry & fishing (% of GDP)",
    "AG.PRD.CREL.MT": "Cereal production (metric tons)",
    "ER.H2O.FWTL.K3": "Annual freshwater withdrawals, total (billion m3)",
}

# Perjanjian/arsitektur multilateral pertanian (WTO AoA dsb.) untuk tab
# "International Agreements on Agricultural Policy"
AGREEMENTS = [
    {
        "name": "WTO Agreement on Agriculture (AoA)",
        "entered": "1995-01-01",
        "type": "Multilateral",
        "members_covered": 164,
        "pillars": ["Market access", "Domestic support", "Export competition"],
        "key_mechanism": "Bound tariffs, TRQs, Amber/Blue/Green Box support caps",
        "dispute_link": "Basis for most agri trade disputes at WTO DSM",
    },
    {
        "name": "WTO Trade Facilitation Agreement (TFA)",
        "entered": "2017-02-22",
        "type": "Multilateral",
        "members_covered": 164,
        "pillars": ["Customs efficiency", "Transparency", "Special & differential treatment"],
        "key_mechanism": "Reduce border delays; linked to LPI customs scores",
        "dispute_link": "Lowers non-tariff friction that triggers SPS complaints",
    },
    {
        "name": "SPS Agreement (Sanitary & Phytosanitary)",
        "entered": "1995-01-01",
        "type": "Multilateral",
        "members_covered": 164,
        "pillars": ["Science-based measures", "Equivalence", "Regionalization"],
        "key_mechanism": "Most-cited agreement in agri-food disputes",
        "dispute_link": "Legal anchor for bans/quotas on agri products",
    },
    {
        "name": "USMCA (US-Mexico-Canada Agreement)",
        "entered": "2020-07-01",
        "type": "Preferential (FTA)",
        "members_covered": 3,
        "pillars": ["Dairy TRQ access", "Biotech labeling", "Rapid response labor"],
        "key_mechanism": "Chapter 3 national treatment; SPS committee provisions",
        "dispute_link": "US-Mexico corn biotech consultation (2023-ongoing)",
    },
    {
        "name": "RCEP (Regional Comprehensive Economic Partnership)",
        "entered": "2022-01-01",
        "type": "Preferential (FTA)",
        "members_covered": 15,
        "pillars": ["Tariff elimination schedule", "Cumulative rules of origin"],
        "key_mechanism": "ASEAN+ agri tariff phase-down toward zero",
        "dispute_link": "Overlap with bilateral sensitivities (rice, sugar)",
    },
    {
        "name": "Indonesia-EU CEPA (signed Dec 2025, ratification pending)",
        "entered": "pending",
        "type": "Preferential (FTA)",
        "members_covered": 2,
        "pillars": ["Tariff removal; palm oil & EUDR carve-outs", "Geographical indications"],
        "key_mechanism": "Negotiated resolution of EU deforestation regulation friction",
        "dispute_link": "Directly addresses market-access barriers cited by agri attachés",
    },
]
