# AgriTrade Pro | APAC Edition - Elite Commodities Trading Intelligence Platform

🌾 **Professional-grade agricultural commodities trading application** with comprehensive APAC coverage, combining machine learning forecasting, economic indicators, seasonal analysis, event-driven shock analysis, and risk management tools.

## Features

### 🔮 ML-Powered Price Forecasting
- Ensemble models (Random Forest, Gradient Boosting, Ridge Regression)
- Time-series cross-validation for robust model evaluation
- Feature importance analysis to understand predictive drivers
- Consensus forecasts from multiple models
- **Configurable forecast base days**: 30, 60, 90, 180, or 365 days

### 📈 Advanced Technical Analysis
- Interactive candlestick charts with volume
- Moving averages (20, 50, 200-day)
- RSI (Relative Strength Index) with overbought/oversold signals
- MACD (Moving Average Convergence Divergence)
- Real-time technical signal summaries
- Buy/sell/hold recommendations with position sizing

### 📅 Seasonal Pattern Analysis
- Historical monthly price patterns
- Average monthly returns visualization
- Seasonal insights for strategic positioning
- Agricultural cycle integration (planting/harvest seasons)
- Time-based trading strategies

### ⚠️ Comprehensive Risk Analytics
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional VaR (CVaR) for tail risk assessment
- Maximum drawdown analysis
- Returns distribution analysis (skewness, kurtosis)
- Normality testing (Jarque-Bera test)
- Q-Q plots for distribution visualization
- Risk-based position sizing and hedging strategies

### 🌍 Event Study Analysis (NEW!)
- **12-month shock impact analysis** across three categories:
  - **Geopolitical Events**: Wars, trade disputes, grain deal uncertainties
  - **Climate Shocks**: El Niño, droughts, floods, monsoon delays
  - **Regulatory Changes**: Export bans, tariffs, ESG regulations, biofuel mandates
  - **Economic Shocks**: Fed decisions, GDP slowdowns, currency movements, inflation
- Interactive price impact timeline with event markers
- Event summary metrics (max negative/positive shocks, average impact)
- Commodity-specific sensitivity multipliers
- Tactical monetization strategies for each event type
- Recovery period analysis

### 💼 Portfolio Optimization
- Multi-commodity portfolio allocation
- Risk-adjusted optimization based on tolerance (conservative/moderate/aggressive)
- Correlation matrix for diversification analysis
- Expected return and volatility calculations
- Sharpe ratio optimization
- Diversification benefit quantification
- Allocation strategies based on performance metrics

### 📊 Supported Commodities

#### Americas Region
- **Grains**: Corn, Wheat, Soybeans
- **Softs**: Coffee, Sugar, Cotton
- **Livestock**: Live Cattle, Lean Hogs

#### APAC Region (NEW!)
- **Grains & Oilseeds**: APAC Rice, APAC Palm Oil
- **Industrial Crops**: APAC Rubber, APAC Cashew, APAC Pepper
- **Beverages**: APAC Tea
- **Energy**: APAC Natural Gas (LNG)
- **Food Products**: APAC Coconut Oil

### 🌐 Regional Filtering
- Filter commodities by region (Americas, APAC, or both)
- Region-specific analysis and insights
- Tailored monetization strategies per geographic market

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

#### Region Selection
- **Select Regions**: Choose Americas, APAC, or both to filter available commodities

#### Commodity Selection
- **Single Commodity Mode**: Select one commodity for detailed analysis
- **Multi-Commodity Mode**: Select multiple commodities for comparison and portfolio optimization

#### Analysis Configuration
- **Analysis Type**: Choose from 6 analysis modes:
  - Technical Analysis
  - ML Forecasting
  - Risk Analytics
  - Seasonal Patterns
  - Portfolio Optimization
  - Event Study (NEW!)
- **Forecast Horizon**: Set prediction window (1-30 days)
- **Forecast Base Days**: Select historical data period (30, 60, 90, 180, or 365 days)
- **Risk Tolerance**: Choose conservative, moderate, or aggressive
- **Refresh Data**: Update price data from markets

## Architecture

### Data Pipeline
1. Real-time price data fetched via Yahoo Finance API
2. Cached for 1 hour to optimize performance
3. Automatic handling of missing data and errors
4. Regional data sourcing for Americas and APAC markets

### Machine Learning Models
- **Random Forest**: 100 trees, max depth 10
- **Gradient Boosting**: 100 estimators, max depth 5
- **Ridge Regression**: L2 regularization (alpha=1.0)
- **TimeSeriesSplit**: 3-fold walk-forward validation
- **Configurable Training Period**: 30-365 days based on user selection

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

### Event Study Framework
- Event identification from global/regional/national shock databases
- Abnormal return calculations around event windows
- Cumulative abnormal return (CAR) aggregation
- Commodity-specific sensitivity multipliers
- Recovery period tracking

## Economic Context Integration

Each commodity includes:
- Trading unit (cents/bushel, cents/lb)
- Seasonal cycles (planting/harvest periods)
- Major producing countries/regions (Americas and APAC)
- Market-specific characteristics
- Regional economic indicators
- Climate sensitivity factors

## Output Metrics

### Price Metrics
- Current price with daily change
- Percentage change indicators
- Volume analysis
- Regional price differentials

### Risk Metrics
- Volatility (annualized %)
- VaR (daily loss threshold)
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown
- CVaR (tail risk)

### Forecast Metrics
- Model R² scores
- Predicted prices with direction
- Consensus forecast
- Feature importance rankings

### Event Study Metrics
- Cumulative Abnormal Returns (CAR)
- Event impact magnitude (%)
- Recovery period (days)
- Sensitivity multipliers
- Shock categorization (geopolitical, climate, regulatory, economic)

### Monetization Strategies
Each analysis type provides specific actionable strategies:
- **Technical Analysis**: Entry/exit signals, position sizing, leverage recommendations
- **ML Forecasting**: Directional positioning based on bullish/bearish/neutral outlooks
- **Risk Analytics**: Hedging strategies, position limits, capital allocation
- **Seasonal Patterns**: Time-based entry/exit strategies
- **Portfolio Optimization**: Optimal weightings, diversification benefits
- **Event Study**: Tactical trades around shocks, event-driven arbitrage

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
- **Event Analysis**: Custom event study framework with shock databases

## Professional Features

Built with expertise from:
- **AI Engineering**: Production-grade ML pipelines
- **Machine Learning**: Ensemble methods, time-series forecasting
- **Agricultural Economics**: Seasonal patterns, supply-demand dynamics
- **Quantitative Finance**: Risk metrics, portfolio theory
- **Event-Driven Strategy**: Shock analysis, tactical trading
- **Regional Markets**: Americas and APAC commodity expertise

## License

See LICENSE file for terms and conditions.

## Support

For questions or issues, please refer to the documentation or contact support.

---

*AgriTrade Pro | APAC Edition - Elite Commodities Trading Intelligence Platform*

*Powered by Advanced Machine Learning • Agricultural Economics • Risk Analytics • Event-Driven Strategies • APAC Market Expertise*
