# AgriTrade Pro - Elite Commodities Trading Intelligence Platform

🌾 **Professional-grade agricultural commodities trading application** combining machine learning forecasting, economic indicators, seasonal analysis, and risk management tools.

## Features

### 🔮 ML-Powered Price Forecasting
- Ensemble models (Random Forest, Gradient Boosting, Ridge Regression)
- Time-series cross-validation for robust model evaluation
- Feature importance analysis to understand predictive drivers
- Consensus forecasts from multiple models

### 📈 Advanced Technical Analysis
- Interactive candlestick charts with volume
- Moving averages (20, 50, 200-day)
- RSI (Relative Strength Index) with overbought/oversold signals
- MACD (Moving Average Convergence Divergence)
- Real-time technical signal summaries

### 📅 Seasonal Pattern Analysis
- Historical monthly price patterns
- Average monthly returns visualization
- Seasonal insights for strategic positioning
- Agricultural cycle integration (planting/harvest seasons)

### ⚠️ Comprehensive Risk Analytics
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional VaR (CVaR) for tail risk assessment
- Maximum drawdown analysis
- Returns distribution analysis (skewness, kurtosis)
- Normality testing (Jarque-Bera test)
- Q-Q plots for distribution visualization

### 💼 Portfolio Optimization
- Multi-commodity portfolio allocation
- Risk-adjusted optimization based on tolerance (conservative/moderate/aggressive)
- Correlation matrix for diversification analysis
- Expected return and volatility calculations
- Sharpe ratio optimization
- Diversification benefit quantification

### 📊 Supported Commodities
- **Grains**: Corn, Wheat, Soybeans
- **Softs**: Coffee, Sugar, Cotton
- **Livestock**: Live Cattle, Lean Hogs

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn yfinance requests scipy
```

## Usage

### Run the Application
```bash
streamlit run commodities_trading_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Control Panel
- **Select Commodities**: Choose one or more commodities to analyze
- **Analysis Type**: Select from 5 analysis modes
- **Forecast Horizon**: Set prediction window (1-30 days)
- **Risk Tolerance**: Choose conservative, moderate, or aggressive
- **Refresh Data**: Update price data from markets

## Architecture

### Data Pipeline
1. Real-time price data fetched via Yahoo Finance API
2. Cached for 1 hour to optimize performance
3. Automatic handling of missing data and errors

### Machine Learning Models
- **Random Forest**: 100 trees, max depth 10
- **Gradient Boosting**: 100 estimators, max depth 5
- **Ridge Regression**: L2 regularization (alpha=1.0)
- **TimeSeriesSplit**: 3-fold walk-forward validation

### Feature Engineering
- Lag features (1, 3, 5, 10, 20 days)
- Rolling statistics (mean, std)
- Momentum indicators
- Volume analysis
- Price position relative to moving averages

### Risk Calculations
- Annualized volatility (252 trading days)
- Historical and parametric VaR
- Rolling window calculations (20-day)
- Statistical tests for distribution properties

## Economic Context Integration

Each commodity includes:
- Trading unit (cents/bushel, cents/lb)
- Seasonal cycles (planting/harvest periods)
- Major producing countries/regions
- Market-specific characteristics

## Output Metrics

### Price Metrics
- Current price with daily change
- Percentage change indicators
- Volume analysis

### Risk Metrics
- Volatility (annualized %)
- VaR (daily loss threshold)
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown

### Forecast Metrics
- Model R² scores
- Predicted prices with direction
- Consensus forecast
- Feature importance rankings

## Disclaimer

⚠️ **This platform provides analytical tools and information for educational and research purposes only.**

- All trading decisions involve substantial risk of loss
- Past performance does not guarantee future results
- Always conduct your own research
- Consult with qualified financial advisors before making investment decisions
- Not intended as financial advice or recommendations

## Technology Stack

- **Frontend**: Streamlit (interactive web interface)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly (interactive charts)
- **Machine Learning**: Scikit-learn
- **Data Source**: Yahoo Finance API
- **Statistics**: SciPy

## Professional Features

Built with expertise from:
- **AI Engineering**: Production-grade ML pipelines
- **Machine Learning**: Ensemble methods, time-series forecasting
- **Agricultural Economics**: Seasonal patterns, supply-demand dynamics
- **Quantitative Finance**: Risk metrics, portfolio theory

## License

See LICENSE file for terms and conditions.

## Support

For questions or issues, please refer to the documentation or contact support.

---

*AgriTrade Pro | Elite Commodities Trading Intelligence Platform*

*Powered by Advanced Machine Learning • Agricultural Economics • Risk Analytics*
