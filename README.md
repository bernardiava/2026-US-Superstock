# 🚀 SuperStock Detector: US Market Alpha Engine (2015-Present)

> **A professional-grade quantitative dashboard for detecting, analyzing, and forecasting "Superstocks" in the US equity market.**  
> *Built with the precision of a CFA charterholder, the rigor of a quant researcher, and the polish of a top-tier fintech designer.*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)

---

## 📖 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack & Skills Demonstrated](#tech-stack--skills-demonstrated)
- [Target Roles](#target-roles)
- [Installation & Usage](#installation--usage)
- [Methodology](#methodology)
- [Disclaimer](#disclaimer)
- [Copyright & Citation](#copyright--citation)

---

## 🌟 Overview

The **SuperStock Detector** is an interactive financial analytics tool designed to identify US stocks that have delivered exponential returns (300%+) since 2015. It goes beyond simple price charts by decomposing stock performance into distinct **lifecycle phases** (Rapid Growth, Stagnation, Decline) and correlating these phases with fundamental events like earnings announcements, stock splits, and macro news.

This project simulates a professional institutional research workflow, combining:
1.  **Fundamental Screening**: Identifying high-growth candidates based on market cap and return thresholds.
2.  **Quantitative Phase Analysis**: Using statistical regimes to detect when a stock is accelerating or stagnating.
3.  **Event-Driven Timeline**: Mapping price action against real-world catalysts.
4.  **Risk Management**: Calculating Sharpe ratios, maximum drawdowns, and volatility regimes.

---

## 🔥 Key Features

### 1. Intelligent SuperStock Detection
- Scans a curated universe of high-growth US equities (e.g., NVDA, TSLA, META, AMD).
- Filters for **300%+ total returns** since 2015.
- Ranks candidates by risk-adjusted returns (Sharpe Ratio) and volatility.

### 2. Lifecycle Phase Decomposition
Automatically segments historical price data into four distinct regimes:
- 🟢 **Rapid Growth**: Annualized returns > 50%.
- 🔵 **Moderate Growth**: Annualized returns 10–50%.
- 🟡 **Stagnation**: Annualized returns -10% to 10% (consolidation phases).
- 🔴 **Decline**: Annualized returns < -10%.

### 3. Interactive Event Timeline
- Visualizes price action overlaid with key corporate events:
  - Earnings Reports (with EPS surprise data).
  - Stock Splits & Dividends.
  - Major News Headlines (via Yahoo Finance integration).
- Allows users to pinpoint exactly *why* a stock moved at a specific time.

### 4. Institutional-Grade UI/UX
- **Dark Mode Interface**: Designed with a Bloomberg Terminal/Two Sigma aesthetic.
- **Real-time Interactivity**: Dynamic filtering, hover tooltips, and cross-linked charts.
- **Performance Optimized**: Parallel processing and caching ensure sub-minute load times for deep historical analysis.

### 5. Quantitative Metrics Dashboard
- **Risk Stats**: Volatility, Max Drawdown, Beta, Skewness, Kurtosis.
- **Trend Indicators**: SMA50/200 crossovers (Golden/Death Cross detection).
- **Return Distribution**: Histograms showing the frequency of daily returns.

---

## 🛠 Tech Stack & Skills Demonstrated

This project showcases a rare hybrid of **Finance**, **Data Science**, and **Software Engineering** skills:

### 💼 Financial Expertise (CFA Level)
- **Equity Research**: Deep understanding of growth drivers, market cycles, and fundamental catalysts.
- **Portfolio Construction**: Selection bias mitigation, sector diversification analysis.
- **Risk Management**: Value at Risk (VaR) concepts, drawdown analysis, and volatility clustering.
- **Market Microstructure**: Understanding of stock splits, dividend impacts, and earnings seasonality.

### 📊 Quantitative Research & Data Science
- **Time Series Analysis**: Regime switching detection, rolling window statistics.
- **Statistical Modeling**: Calculation of higher-order moments (skewness, kurtosis) for return distributions.
- **Data Engineering**: Efficient ETL pipelines using `yfinance`, handling missing data, and normalizing multi-asset time series.
- **Forecasting Readiness**: Architecture designed to integrate ARIMA, GARCH, LSTM, or Prophet models for future price projection.

### 💻 Full-Stack Development & UI/UX
- **Python Proficiency**: Advanced use of `pandas`, `numpy`, `scipy`, and `parallel processing`.
- **Interactive Dashboards**: Expert-level `Streamlit` development (session state management, caching strategies, custom layouts).
- **Data Visualization**: Mastery of `Plotly` for creating publication-quality financial charts (candlesticks, area charts, histograms).
- **UX Design**: Intuitive user flows, responsive layout design, and professional color theory application for financial data.

---

## 🎯 Target Roles

This portfolio project is tailored to demonstrate competency for the following roles:

| Role | Relevance |
| :--- | :--- |
| **Quantitative Researcher** | Demonstrates ability to handle large time-series datasets, define alpha factors, and analyze market regimes statistically. |
| **Equity Research Analyst** | Shows capability to link price action to fundamental events (earnings, news) and perform deep-dive due diligence. |
| **Financial Data Scientist** | Highlights end-to-end pipeline creation: from raw data ingestion to interactive visualization and statistical inference. |
| **Portfolio Manager / Strategist** | Illustrates risk-aware investment thinking, drawdown control, and sector rotation analysis. |
| **FinTech Product Engineer** | Proves ability to build user-centric, high-performance financial tools with professional-grade UI/UX. |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy plotly yfinance scipy tqdm
```

### 3. Run the Application
```bash
streamlit run superstock_detector.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

### 4. How to Use
1.  **Sidebar**: Select a specific stock ticker or analyze the "Top Superstocks" list.
2.  **Overview Tab**: View the ranking table and sector distribution.
3.  **Deep Dive Tab**: Analyze specific growth phases and stagnation periods.
4.  **Timeline Tab**: Explore the interactive chart with event markers (earnings, splits, news).
5.  **Metrics Tab**: Review detailed statistical properties and risk metrics.

---

## 🧠 Methodology

1.  **Data Ingestion**: Historical daily OHLCV data is fetched via `yfinance` from Jan 1, 2015, to present.
2.  **Screening**: Stocks are filtered for minimum market cap and total return thresholds.
3.  **Regime Detection**: A rolling annualized return calculation classifies each day into a specific phase (Growth/Stagnation/Decline).
4.  **Event Correlation**: Earnings dates and split events are merged with price data to identify catalysts.
5.  **Visualization**: Data is rendered using Plotly with custom themes to ensure clarity and professional presentation.

---

## ⚠️ Disclaimer

*This application is for **educational and research purposes only**. It does not constitute financial advice, investment recommendations, or an offer to sell or buy any securities. Past performance is not indicative of future results. The author makes no representations as to the accuracy, completeness, or timeliness of the data. Always conduct your own due diligence before making investment decisions.*

---

## © Copyright & Citation

**All Rights Reserved.**  
© 2024-2025. This codebase, design, and analytical methodology are the exclusive property of the author. Unauthorized commercial use, redistribution, or claiming of this work as your own is strictly prohibited.

### 📚 Citation
If you find this project inspiring or use it as a reference for your own research, portfolio, or academic work, please cite it as follows:

> **Bernardia Vitri Arumsari**. (2025). *SuperStock Detector: US Market Alpha Engine*. GitHub Repository. Available at: [https://github.com/bernardiava/2026-Various-Projects/tree/superstocksusmarketanalysis-9b6fe]

*Acknowledging the original author helps support the open-source community and encourages further development of high-quality financial tools.*

---

*Built with ❤️ by a team of Finance, Quant, and Design experts.*
