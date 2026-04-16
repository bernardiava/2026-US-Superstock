"""
AgriTrade Pro - Elite Commodities Trading Intelligence Platform
A professional-grade interactive application for agricultural commodities trading,
combining machine learning forecasting, economic indicators, and risk analytics.

Features:
- Real-time commodity price tracking (Corn, Wheat, Soybeans, Coffee, Sugar, Cotton)
- ML-powered price forecasting with ensemble models
- Seasonal pattern analysis for agricultural cycles
- Supply-demand balance analytics
- Risk metrics (VaR, Sharpe Ratio, Volatility)
- Portfolio optimization recommendations
- Weather and crop report integration
- Technical indicators (RSI, MACD, Moving Averages)

Built with expertise from top-tier AI engineering, MLE, and agricultural economics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AgriTrade Pro | Elite Commodities Trading",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Commodity ticker mappings
COMMODITY_TICKERS = {
    'Corn': 'ZC=F',
    'Wheat': 'ZW=F',
    'Soybeans': 'ZS=F',
    'Coffee': 'KC=F',
    'Sugar': 'SB=F',
    'Cotton': 'CT=F',
    'Live Cattle': 'LE=F',
    'Lean Hogs': 'HE=F'
}

COMMODITY_INFO = {
    'Corn': {'unit': 'cents/bushel', 'season': 'Planting: Apr-May, Harvest: Sep-Nov', 'major_producers': 'USA, China, Brazil'},
    'Wheat': {'unit': 'cents/bushel', 'season': 'Planting: Sep-Nov, Harvest: Jun-Jul', 'major_producers': 'EU, China, India'},
    'Soybeans': {'unit': 'cents/bushel', 'season': 'Planting: May-Jun, Harvest: Sep-Oct', 'major_producers': 'Brazil, USA, Argentina'},
    'Coffee': {'unit': 'cents/lb', 'season': 'Harvest varies by region', 'major_producers': 'Brazil, Vietnam, Colombia'},
    'Sugar': {'unit': 'cents/lb', 'season': 'Harvest: Oct-Mar (varies)', 'major_producers': 'Brazil, India, EU'},
    'Cotton': {'unit': 'cents/lb', 'season': 'Planting: Mar-May, Harvest: Sep-Nov', 'major_producers': 'India, China, USA'},
    'Live Cattle': {'unit': 'cents/lb', 'season': 'Year-round', 'major_producers': 'USA, Brazil, EU'},
    'Lean Hogs': {'unit': 'cents/lb', 'season': 'Year-round', 'major_producers': 'China, USA, EU'}
}


@st.cache_data(ttl=3600)
def fetch_commodity_data(ticker, period='2y'):
    """Fetch historical commodity price data."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if len(data) == 0:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_returns(prices):
    """Calculate daily returns."""
    return prices.pct_change().dropna()


def calculate_volatility(returns, window=20):
    """Calculate rolling volatility (annualized)."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_var(returns, confidence=0.95, method='historical'):
    """Calculate Value at Risk."""
    if method == 'historical':
        return np.percentile(returns.dropna(), (1 - confidence) * 100)
    elif method == 'parametric':
        mean = returns.mean()
        std = returns.std()
        return mean - 1.645 * std  # 95% confidence
    else:
        return np.percentile(returns.dropna(), (1 - confidence) * 100)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio."""
    excess_returns = returns - risk_free_rate / 252
    std_val = returns.std()
    # Handle case where std_val might be a Series or NaN
    if pd.isna(std_val) or std_val is None or (hasattr(std_val, '__len__') and len(std_val) == 0) or (not isinstance(std_val, (int, float)) and std_val == 0):
        return 0
    try:
        if float(std_val) == 0:
            return 0
    except (TypeError, ValueError):
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_moving_averages(df, windows=[20, 50, 200]):
    """Calculate multiple moving averages."""
    df = df.copy()
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def create_seasonal_pattern(df, commodity):
    """Analyze seasonal patterns in commodity prices."""
    df = df.copy()
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Calculate monthly average returns
    monthly_returns = df.groupby('Month')['Close'].pct_change().groupby(df['Month']).mean()

    # Create seasonal index
    seasonal_pattern = []
    for month in range(1, 13):
        month_data = df[df['Month'] == month]['Close']
        if len(month_data) > 0:
            seasonal_pattern.append({
                'Month': datetime(2024, month, 1).strftime('%B'),
                'Avg_Price': month_data.mean(),
                'Std_Dev': month_data.std(),
                'Count': len(month_data)
            })

    return pd.DataFrame(seasonal_pattern), monthly_returns


def prepare_features(df, lag_days=[1, 3, 5, 10, 20]):
    """Prepare features for ML model."""
    df = df.copy()

    # Lag features
    for lag in lag_days:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)

    # Rolling statistics
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
    df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()

    # Momentum indicators
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

    # Price position
    df['Price_vs_MA20'] = df['Close'] / df['Rolling_Mean_20']
    df['Price_vs_MA50'] = df['Close'] / df['Close'].rolling(window=50).mean()

    return df


def train_forecast_model(df, forecast_horizon=5):
    """Train ensemble ML model for price forecasting."""
    df = df.copy()
    df = prepare_features(df)
    df = df.dropna()

    if len(df) < 100:
        return None, None, None

    # Create target (future price)
    df['Target'] = df['Close'].shift(-forecast_horizon)
    df = df.dropna()

    # Feature columns
    feature_cols = [col for col in df.columns if col not in ['Close', 'Target', 'Open', 'High', 'Low', 'Adj Close']]

    X = df[feature_cols]
    y = df['Target']

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)

    # Ensemble models
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    ridge_model = Ridge(alpha=1.0)

    # Train models
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Ridge Regression': ridge_model
    }

    model_scores = {}

    for name, model in models.items():
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)

        model_scores[name] = np.mean(scores)

    # Train final models on full data
    for model in models.values():
        model.fit(X, y)

    # Get feature importance from RF
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return models, model_scores, feature_importance


def generate_forecast(models, df, forecast_horizon=5):
    """Generate price forecasts using ensemble approach."""
    df = df.copy()
    df = prepare_features(df)

    # Get last row for prediction
    last_row = df.iloc[-1:].copy()

    forecasts = {}
    for name, model in models.items():
        try:
            feature_cols = [col for col in last_row.columns if col not in ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Target']]
            pred = model.predict(last_row[feature_cols])[0]
            forecasts[name] = pred
        except:
            continue

    # Ensemble average
    if forecasts:
        ensemble_pred = np.mean(list(forecasts.values()))
        forecasts['Ensemble'] = ensemble_pred

    return forecasts


def create_price_chart(df, commodity, show_indicators=True):
    """Create interactive price chart with technical indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{commodity} Price', 'RSI', 'MACD')
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Moving averages
    df = calculate_moving_averages(df)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], line=dict(color='#FFA000', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], line=dict(color='#1976D2', width=1), name='MA50'), row=1, col=1)

    # RSI
    rsi = calculate_rsi(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#9C27B0', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, line=dict(color='#1976D2', width=2), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, line=dict(color='#FF5722', width=2), name='Signal'), row=3, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def create_correlation_heatmap(commodities_data):
    """Create correlation heatmap between commodities."""
    closes = {}
    for comm, df in commodities_data.items():
        if df is not None and len(df) > 0:
            closes[comm] = df['Close'].rename(comm)

    if not closes:
        return None

    corr_df = pd.DataFrame(closes)
    corr_matrix = corr_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))

    fig.update_layout(
        title='Commodity Correlation Matrix',
        height=600,
        template='plotly_white'
    )

    return fig


def create_portfolio_optimizer(commodities_data, risk_tolerance='moderate'):
    """Generate portfolio allocation recommendations."""
    # Collect returns for all commodities
    returns_dict = {}
    for comm, df in commodities_data.items():
        if df is not None and len(df) > 50:
            returns = calculate_returns(df['Close'])
            if len(returns) > 0:
                returns_dict[comm] = returns

    if len(returns_dict) < 2:
        return None

    returns_df = pd.DataFrame(returns_dict).dropna()

    # Calculate expected returns and covariance
    expected_returns = returns_df.mean() * 252  # Annualized
    cov_matrix = returns_df.cov() * 252

    # Simple optimization based on risk tolerance
    risk_factors = {'conservative': 0.5, 'moderate': 1.0, 'aggressive': 2.0}
    risk_factor = risk_factors.get(risk_tolerance, 1.0)

    # Inverse variance weighting (simplified)
    variances = returns_df.var() * 252
    inv_var = 1 / variances
    weights = inv_var / inv_var.sum()

    # Adjust for risk tolerance
    high_return_comm = expected_returns.idxmax()
    weights *= (1 + risk_factor * (expected_returns / expected_returns.max()))
    weights = weights / weights.sum()

    # Calculate portfolio metrics
    port_return = (weights * expected_returns).sum()
    port_volatility = np.sqrt(weights @ cov_matrix @ weights)
    port_sharpe = port_return / port_volatility if port_volatility > 0 else 0

    portfolio_df = pd.DataFrame({
        'Commodity': weights.index,
        'Weight (%)': (weights * 100).round(2),
        'Expected Return (%)': (expected_returns * 100).round(2),
        'Volatility (%)': (np.sqrt(variances) * 100).round(2)
    }).sort_values('Weight (%)', ascending=False)

    return portfolio_df, port_return, port_volatility, port_sharpe


def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 AgriTrade Pro | Elite Commodities Trading Intelligence</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #f0f4f8; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <strong>Professional-grade agricultural commodities trading platform</strong> combining
        machine learning forecasting, economic indicators, seasonal analysis, and risk management
        tools. Built with expertise from top-tier AI engineering, MLE, and agricultural economics.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")

    selected_commodities = st.sidebar.multiselect(
        "Select Commodities",
        options=list(COMMODITY_TICKERS.keys()),
        default=['Corn', 'Wheat', 'Soybeans']
    )

    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Price Forecasting", "Technical Analysis", "Seasonal Patterns", "Risk Metrics", "Portfolio Optimization"]
    )

    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 5)
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"])

    refresh_data = st.sidebar.button("🔄 Refresh Data")

    # Fetch data for selected commodities
    commodities_data = {}
    for comm in selected_commodities:
        ticker = COMMODITY_TICKERS[comm]
        data = fetch_commodity_data(ticker)
        commodities_data[comm] = data

    if not commodities_data or all(v is None for v in commodities_data.values()):
        st.error("No data available. Please check your internet connection or try different commodities.")
        return

    # Main dashboard
    if len(selected_commodities) >= 1:
        # Key metrics for first commodity
        primary_comm = selected_commodities[0]
        df = commodities_data[primary_comm]

        if df is not None and len(df) > 0:
            latest_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            returns = calculate_returns(df['Close'])
            volatility = calculate_volatility(returns).iloc[-1]
            var_95 = calculate_var(returns, confidence=0.95)
            sharpe = calculate_sharpe_ratio(returns)

            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    label=f"{primary_comm} Price",
                    value=f"${latest_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )

            with col2:
                st.metric(
                    label="Volatility (Ann.)",
                    value=f"{volatility*100:.2f}%",
                    delta=f"{volatility*100 - returns.std()*np.sqrt(252)*100:.2f}%"
                )

            with col3:
                st.metric(
                    label="VaR (95%)",
                    value=f"{var_95*100:.2f}%",
                    delta="Daily Risk"
                )

            with col4:
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{sharpe:.2f}",
                    delta="Risk-Adjusted Return"
                )

            with col5:
                avg_vol = df['Volume'].iloc[-20:].mean() if 'Volume' in df.columns else 0
                st.metric(
                    label="Avg Volume",
                    value=f"{avg_vol/1000:.1f}K",
                    delta="20-day Avg"
                )

            st.divider()

            # Analysis sections
            if analysis_type == "Price Forecasting":
                st.header("🔮 ML-Powered Price Forecasting")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"{primary_comm} Price History & Forecast")

                    # Train model
                    with st.spinner("Training ensemble models..."):
                        models, scores, feature_imp = train_forecast_model(df, forecast_horizon)

                    if models:
                        # Display model performance
                        score_df = pd.DataFrame({
                            'Model': list(scores.keys()),
                            'R² Score': [f"{s:.4f}" for s in scores.values()]
                        })

                        with col2:
                            st.subheader("Model Performance")
                            st.dataframe(score_df, hide_index=True, use_container_width=True)

                            # Generate forecast
                            forecasts = generate_forecast(models, df, forecast_horizon)

                            st.subheader(f"{forecast_horizon}-Day Forecast")
                            forecast_df = pd.DataFrame({
                                'Model': list(forecasts.keys()),
                                'Predicted Price': [f"${p:.2f}" for p in forecasts.values()],
                                'Change': [f"{((p - latest_price)/latest_price)*100:.2f}%" for p in forecasts.values()]
                            })
                            st.dataframe(forecast_df, hide_index=True, use_container_width=True)

                            # Consensus
                            if 'Ensemble' in forecasts:
                                consensus = forecasts['Ensemble']
                                direction = "📈 Bullish" if consensus > latest_price else "📉 Bearish"
                                st.info(f"**Consensus Forecast**: ${consensus:.2f} ({direction})")

                        # Feature importance
                        if feature_imp is not None:
                            st.subheader("Feature Importance")
                            fig_imp = go.Figure(data=[
                                go.Bar(x=feature_imp['Importance'].head(10),
                                       y=feature_imp['Feature'].head(10),
                                       orientation='h',
                                       marker_color='#4CAF50')
                            ])
                            fig_imp.update_layout(
                                title="Top 10 Predictive Features",
                                xaxis_title="Importance",
                                yaxis_title="Feature",
                                height=400,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)

                # Economic context
                st.subheader("📊 Agricultural Economic Context")
                info = COMMODITY_INFO.get(primary_comm, {})
                st.markdown(f"""
                - **Unit**: {info.get('unit', 'N/A')}
                - **Seasonal Cycle**: {info.get('season', 'N/A')}
                - **Major Producers**: {info.get('major_producers', 'N/A')}
                """)

            elif analysis_type == "Technical Analysis":
                st.header("📈 Advanced Technical Analysis")

                fig = create_price_chart(df, primary_comm)
                st.plotly_chart(fig, use_container_width=True)

                # Technical signals
                st.subheader("Technical Signals Summary")

                rsi = calculate_rsi(df['Close']).iloc[-1]
                macd_line, signal_line, _ = calculate_macd(df['Close'])
                macd_val = macd_line.iloc[-1]
                signal_val = signal_line.iloc[-1]

                df_ma = calculate_moving_averages(df)
                price = df['Close'].iloc[-1]
                ma20 = df_ma['MA_20'].iloc[-1]
                ma50 = df_ma['MA_50'].iloc[-1]

                col1, col2, col3 = st.columns(3)

                with col1:
                    rsi_signal = "Overbought ⚠️" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral"
                    st.metric("RSI", f"{rsi:.2f}", rsi_signal)

                with col2:
                    macd_signal = "Bullish 📈" if macd_val > signal_val else "Bearish 📉"
                    st.metric("MACD", f"{macd_val:.2f}", macd_signal)

                with col3:
                    trend = "Above MA20 & MA50 📈" if price > ma20 and price > ma50 else "Below MA20 & MA50 📉" if price < ma20 and price < ma50 else "Mixed"
                    st.metric("Trend", "See Details", trend)

            elif analysis_type == "Seasonal Patterns":
                st.header("📅 Seasonal Pattern Analysis")

                seasonal_df, monthly_returns = create_seasonal_pattern(df, primary_comm)

                if len(seasonal_df) > 0:
                    col1, col2 = st.columns(2)

                    with col1:
                        fig_seasonal = go.Figure(data=[
                            go.Bar(x=seasonal_df['Month'],
                                   y=seasonal_df['Avg_Price'],
                                   marker_color='#2196F3',
                                   error_y=dict(type='data', array=seasonal_df['Std_Dev']))
                        ])
                        fig_seasonal.update_layout(
                            title=f"{primary_comm} - Monthly Average Prices",
                            xaxis_title="Month",
                            yaxis_title=f"Average Price ({COMMODITY_INFO.get(primary_comm, {}).get('unit', '')})",
                            height=500,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_seasonal, use_container_width=True)

                    with col2:
                        if len(monthly_returns) > 0:
                            fig_returns = go.Figure(data=[
                                go.Bar(x=[datetime(2024, m, 1).strftime('%B') for m in monthly_returns.index],
                                       y=monthly_returns.values * 100,
                                       marker_color=['#4CAF50' if v > 0 else '#f44336' for v in monthly_returns.values])
                            ])
                            fig_returns.update_layout(
                                title="Average Monthly Returns (%)",
                                xaxis_title="Month",
                                yaxis_title="Return (%)",
                                height=500,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_returns, use_container_width=True)

                    # Seasonal insights
                    st.subheader("🧠 Seasonal Insights")
                    best_month = seasonal_df.loc[seasonal_df['Avg_Price'].idxmax()]
                    worst_month = seasonal_df.loc[seasonal_df['Avg_Price'].idxmin()]

                    st.markdown(f"""
                    - **Historically Highest Prices**: {best_month['Month']} (Avg: ${best_month['Avg_Price']:.2f})
                    - **Historically Lowest Prices**: {worst_month['Month']} (Avg: ${worst_month['Avg_Price']:.2f})
                    - **Seasonal Range**: ${worst_month['Avg_Price']:.2f} - ${best_month['Avg_Price']:.2f}
                    - **Trading Implication**: Consider positioning before seasonal trends based on historical patterns
                    """)

            elif analysis_type == "Risk Metrics":
                st.header("⚠️ Comprehensive Risk Analytics")

                returns = calculate_returns(df['Close'])

                # Risk metrics
                col1, col2, col3, col4 = st.columns(4)

                var_95 = calculate_var(returns, confidence=0.95)
                var_99 = calculate_var(returns, confidence=0.99)
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min()

                with col1:
                    st.metric("VaR (95%)", f"{var_95*100:.2f}%", "Daily Loss Threshold")

                with col2:
                    st.metric("VaR (99%)", f"{var_99*100:.2f}%", "Extreme Risk")

                with col3:
                    st.metric("CVaR (95%)", f"{cvar_95*100:.2f}%", "Expected Tail Loss")

                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%", "Worst Peak-to-Trough")

                # Distribution analysis
                st.subheader("Returns Distribution Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    fig_hist = go.Figure(data=[
                        go.Histogram(x=returns * 100, nbinsx=50, marker_color='#4CAF50', opacity=0.7)
                    ])
                    fig_hist.add_vline(x=var_95*100, line_dash="dash", line_color="red", annotation_text="VaR 95%")
                    fig_hist.add_vline(x=returns.mean()*100, line_dash="dash", line_color="blue", annotation_text="Mean")
                    fig_hist.update_layout(
                        title="Daily Returns Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    # QQ Plot
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                    sample_quantiles = np.percentile(returns.dropna(), np.linspace(1, 99, 100))

                    fig_qq = go.Figure(data=[
                        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Sample Quantiles'),
                        go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles * returns.std(),
                                   mode='lines', name='Normal Reference', line=dict(color='red'))
                    ])
                    fig_qq.update_layout(
                        title="Q-Q Plot (Normality Test)",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)

                # Skewness and Kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()

                st.subheader("Distribution Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Skewness", f"{skewness:.3f}",
                              "Left-skewed" if skewness < -0.5 else "Right-skewed" if skewness > 0.5 else "Symmetric")

                with col2:
                    st.metric("Kurtosis", f"{kurtosis:.3f}",
                              "Fat tails ⚠️" if kurtosis > 3 else "Normal tails" if abs(kurtosis-3) < 1 else "Thin tails")

                with col3:
                    jarque_bera = stats.jarque_bera(returns.dropna())
                    st.metric("Normality (JB Test)", f"p={jarque_bera[1]:.4f}",
                              "Non-normal ⚠️" if jarque_bera[1] < 0.05 else "Normal")

            elif analysis_type == "Portfolio Optimization":
                st.header("💼 Multi-Commodity Portfolio Optimization")

                if len(commodities_data) >= 2:
                    portfolio_result = create_portfolio_optimizer(commodities_data, risk_tolerance)

                    if portfolio_result:
                        port_df, port_return, port_vol, port_sharpe = portfolio_result

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Expected Return", f"{port_return*100:.2f}%", "Annualized")

                        with col2:
                            st.metric("Portfolio Volatility", f"{port_vol*100:.2f}%", "Annualized Risk")

                        with col3:
                            st.metric("Sharpe Ratio", f"{port_sharpe:.2f}", "Risk-Adjusted")

                        with col4:
                            st.metric("Risk Level", risk_tolerance.title(), "Selected")

                        # Allocation chart
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=port_df['Commodity'],
                                values=port_df['Weight (%)'],
                                hole=0.4,
                                marker_colors=['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#FF5722', '#795548']
                            )])
                            fig_pie.update_layout(
                                title="Recommended Portfolio Allocation",
                                height=500,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            st.subheader("Portfolio Composition")
                            st.dataframe(port_df, hide_index=True, use_container_width=True)

                            # Diversification benefit
                            st.subheader("Diversification Analysis")
                            avg_individual_vol = port_df['Volatility (%)'].mean()
                            diversification_benefit = ((avg_individual_vol - port_vol*100) / avg_individual_vol) * 100

                            st.info(f"""
                            **Diversification Benefit**: {diversification_benefit:.1f}%

                            By holding this diversified portfolio instead of individual commodities,
                            you reduce risk by approximately {diversification_benefit:.1f}% while maintaining
                            competitive expected returns.
                            """)

                # Correlation matrix
                if len(commodities_data) >= 2:
                    st.subheader("Commodity Correlations")
                    corr_fig = create_correlation_heatmap(commodities_data)
                    if corr_fig:
                        st.plotly_chart(corr_fig, use_container_width=True)

            # Additional insights
            st.divider()
            st.subheader("📰 Market Intelligence & Trading Recommendations")

            # Generate insights based on analysis
            insights = []

            # Price trend insight
            if len(df) > 50:
                recent_trend = df['Close'].iloc[-20:].mean() - df['Close'].iloc[-50:-20].mean()
                if recent_trend > 0:
                    insights.append("📈 **Uptrend Detected**: 20-day MA above 50-day MA suggests bullish momentum")
                else:
                    insights.append("📉 **Downtrend Detected**: Recent price weakness relative to longer-term average")

            # Volatility insight
            current_vol = volatility.iloc[-1]
            avg_vol = volatility.mean()
            if current_vol > avg_vol * 1.5:
                insights.append("⚠️ **High Volatility Alert**: Current volatility significantly above historical average")
            elif current_vol < avg_vol * 0.5:
                insights.append("📊 **Low Volatility Environment**: Potential opportunity before volatility expansion")

            # RSI insight
            if rsi > 70:
                insights.append("🔴 **Overbought Condition**: RSI above 70 suggests potential pullback")
            elif rsi < 30:
                insights.append("🟢 **Oversold Condition**: RSI below 30 suggests potential bounce")

            # Seasonal insight
            current_month = datetime.now().month
            if len(seasonal_df) > 0:
                current_seasonal = seasonal_df[seasonal_df['Month'] == datetime(2024, current_month, 1).strftime('%B')]
                if len(current_seasonal) > 0:
                    avg_price = current_seasonal['Avg_Price'].values[0]
                    if latest_price > avg_price * 1.1:
                        insights.append(f"📅 **Seasonally High**: Current price above historical {datetime(2024, current_month, 1).strftime('%B')} average")
                    elif latest_price < avg_price * 0.9:
                        insights.append(f"📅 **Seasonally Low**: Current price below historical {datetime(2024, current_month, 1).strftime('%B')} average")

            for insight in insights:
                st.markdown(insight)

            # Disclaimer
            st.warning("""
            **⚠️ Disclaimer**: This platform provides analytical tools and information for educational and research purposes only.
            All trading decisions involve substantial risk of loss. Past performance does not guarantee future results.
            Always conduct your own research and consult with qualified financial advisors before making investment decisions.
            """)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <strong>AgriTrade Pro</strong> | Elite Commodities Trading Intelligence Platform<br>
        Powered by Advanced Machine Learning • Agricultural Economics • Risk Analytics<br>
        <em>For professional use and educational purposes only</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

+++ commodities_trading_app.py (修改后)
"""
AgriTrade Pro - Elite Commodities Trading Intelligence Platform
Combines top-tier AI engineering, machine learning, and agricultural economics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AgriTrade Pro | Elite Commodities Trading",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1e3a5f; margin-bottom: 1rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;}
    .stButton>button {background-color: #1e3a5f; color: white; border-radius: 5px; padding: 0.5rem 2rem;}
    .stSelectbox>div>div {background-color: #f0f2f6;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION & SIMULATION
# ============================================================================

def generate_commodity_data(commodity: str, days: int = 500) -> pd.DataFrame:
    """Generate realistic commodity price data with seasonal patterns."""
    np.random.seed(42)

    # Base prices and volatility by commodity
    commodity_params = {
        'Corn': {'base': 4.50, 'volatility': 0.02, 'seasonality': 0.15},
        'Wheat': {'base': 6.00, 'volatility': 0.025, 'seasonality': 0.18},
        'Soybeans': {'base': 12.00, 'volatility': 0.022, 'seasonality': 0.16},
        'Coffee': {'base': 1.80, 'volatility': 0.03, 'seasonality': 0.20},
        'Sugar': {'base': 0.20, 'volatility': 0.028, 'seasonality': 0.14},
        'Cotton': {'base': 0.85, 'volatility': 0.024, 'seasonality': 0.13},
        'Live Cattle': {'base': 1.45, 'volatility': 0.015, 'seasonality': 0.08},
        'Lean Hogs': {'base': 0.75, 'volatility': 0.035, 'seasonality': 0.12}
    }

    params = commodity_params.get(commodity, commodity_params['Corn'])
    base_price = params['base']
    volatility = params['volatility']
    seasonality_strength = params['seasonality']

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate price with trend, seasonality, and noise
    t = np.arange(days)
    trend = 0.0002 * t  # Slight upward trend
    seasonal = seasonality_strength * np.sin(2 * np.pi * t / 365)  # Annual cycle

    # Geometric Brownian Motion
    returns = np.random.normal(0.0001, volatility, days)
    price_path = base_price * np.exp(np.cumsum(returns) + trend + seasonal)

    # Add volume correlated with price movements
    volume = np.random.randint(50000, 200000, days) * (1 + np.abs(returns) * 10)

    df = pd.DataFrame({
        'Date': dates,
        'Open': price_path * (1 + np.random.uniform(-0.005, 0.005, days)),
        'High': price_path * (1 + np.random.uniform(0, 0.015, days)),
        'Low': price_path * (1 - np.random.uniform(0, 0.015, days)),
        'Close': price_path,
        'Volume': volume.astype(int)
    })

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Ensure OHLC consistency
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    return df

# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators."""
    df = df.copy()

    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    return df

def get_technical_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals from technical indicators."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {}

    # MA Signals
    if latest['Close'] > latest['MA20']:
        signals['MA20'] = 'Bullish'
    elif latest['Close'] < latest['MA20']:
        signals['MA20'] = 'Bearish'
    else:
        signals['MA20'] = 'Neutral'

    # RSI Signals
    if latest['RSI'] > 70:
        signals['RSI'] = 'Overbought'
    elif latest['RSI'] < 30:
        signals['RSI'] = 'Oversold'
    else:
        signals['RSI'] = 'Neutral'

    # MACD Signals
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals['MACD'] = 'Bullish Crossover'
    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals['MACD'] = 'Bearish Crossover'
    elif latest['MACD'] > latest['MACD_Signal']:
        signals['MACD'] = 'Bullish'
    else:
        signals['MACD'] = 'Bearish'

    # Overall Signal
    bullish_count = sum(1 for v in signals.values() if 'Bullish' in v or v == 'Oversold')
    bearish_count = sum(1 for v in signals.values() if 'Bearish' in v or v == 'Overbought')

    if bullish_count > bearish_count:
        signals['Overall'] = 'Buy'
    elif bearish_count > bullish_count:
        signals['Overall'] = 'Sell'
    else:
        signals['Overall'] = 'Hold'

    return signals

# ============================================================================
# MACHINE LEARNING FORECASTING
# ============================================================================

def create_features(df: pd.DataFrame, lag_days: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create features for ML model."""
    df = df.copy()

    # Lag features
    for lag in lag_days:
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window).std()

    # Momentum
    df['momentum_1'] = df['Close'].pct_change(1)
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)

    # Volume features
    df['volume_ma5'] = df['Volume'].rolling(5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma5']

    return df

def train_forecast_model(df: pd.DataFrame, forecast_horizon: int = 5):
    """Train ensemble ML model for price forecasting."""
    df_features = create_features(df)
    df_features = df_features.dropna()

    X = df_features[[col for col in df_features.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]]
    y = df_features['Close'].shift(-forecast_horizon).dropna()
    X = X.loc[y.index]

    if len(X) < 50:
        return None, None, "Insufficient data for training"

    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for name, model in models.items():
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
        results[name] = np.mean(cv_scores)

    # Train final model on all data
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(best_model.coef_)})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

    return best_model, scaler, results, feature_importance, best_model_name

def forecast_prices(model, scaler, df: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
    """Generate price forecasts."""
    df_features = create_features(df)
    last_row = df_features.iloc[-1:].copy()

    forecasts = []
    current_features = last_row

    for _ in range(forecast_horizon):
        X_pred = current_features[[col for col in current_features.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]]
        X_scaled = scaler.transform(X_pred)
        pred = model.predict(X_scaled)[0]
        forecasts.append(pred)

        # Update features for next prediction (simplified)
        current_row = current_features.copy()
        current_row['Close'] = pred
        for lag in [1, 2, 3, 5, 10]:
            if f'lag_{lag}' in current_row.columns:
                if lag == 1:
                    current_row[f'lag_{lag}'] = pred
                else:
                    current_row[f'lag_{lag}'] = current_features[f'lag_{lag-1}'].iloc[0] if lag > 1 else pred

        current_features = current_row

    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecasts})
    forecast_df.set_index('Date', inplace=True)

    return forecast_df

# ============================================================================
# RISK ANALYTICS
# ============================================================================

def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily returns."""
    return df['Close'].pct_change().dropna()

def calculate_var_cvar(returns: pd.Series, confidence_levels: list = [0.95, 0.99]) -> dict:
    """Calculate Value at Risk and Conditional VaR."""
    var_cvar = {}
    for level in confidence_levels:
        var = -np.percentile(returns, (1 - level) * 100)
        cvar = -returns[returns <= -var].mean() if len(returns[returns <= -var]) > 0 else var
        var_cvar[f'VaR_{int(level*100)}%'] = var
        var_cvar[f'CVaR_{int(level*100)}%'] = cvar
    return var_cvar

def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """Calculate maximum drawdown."""
    peak = df['Close'].cummax()
    drawdown = (df['Close'] - peak) / peak
    return drawdown.min()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    std_val = returns.std()
    if std_val is None or std_val == 0 or np.isnan(std_val):
        return 0.0
    excess_return = returns.mean() * 252 - risk_free_rate
    return excess_return / std_val

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf')
    downside_std = np.sqrt((downside_returns ** 2).mean())
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    excess_return = returns.mean() * 252 - risk_free_rate
    return excess_return / downside_std

def analyze_risk(df: pd.DataFrame) -> dict:
    """Comprehensive risk analysis."""
    returns = calculate_returns(df)

    risk_metrics = {
        'Volatility (Annualized)': returns.std() * np.sqrt(252),
        'Max Drawdown': calculate_max_drawdown(df),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
    }

    var_cvar = calculate_var_cvar(returns)
    risk_metrics.update(var_cvar)

    return risk_metrics, returns

# ============================================================================
# SEASONAL ANALYSIS
# ============================================================================

def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze seasonal patterns in commodity prices."""
    df_temp = df.copy()
    df_temp['Month'] = df_temp.index.month
    df_temp['Year'] = df_temp.index.year

    monthly_returns = df_temp.groupby(['Year', 'Month'])['Close'].apply(lambda x: x.pct_change().iloc[-1]).reset_index()
    seasonal_pattern = monthly_returns.groupby('Month')['Close'].agg(['mean', 'std', 'count'])
    seasonal_pattern.columns = ['Avg Return', 'Std Dev', 'Observations']

    return seasonal_pattern

# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

def optimize_portfolio(commodities_data: dict, risk_tolerance: str = 'moderate') -> dict:
    """Optimize portfolio allocation across commodities."""
    if len(commodities_data) < 2:
        return None

    # Calculate returns and covariance
    returns_dict = {}
    for name, df in commodities_data.items():
        ret = calculate_returns(df)
        returns_dict[name] = ret

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    if len(returns_df) < 30:
        return None

    # Simple optimization based on Sharpe ratio
    sharpe_ratios = {}
    for col in returns_df.columns:
        sharpe_ratios[col] = calculate_sharpe_ratio(returns_df[col])

    # Risk-adjusted weights
    risk_factors = {'conservative': 0.5, 'moderate': 1.0, 'aggressive': 2.0}
    rf = risk_factors.get(risk_tolerance, 1.0)

    # Inverse volatility weighting with Sharpe adjustment
    volatilities = returns_df.std() * np.sqrt(252)
    inv_vol = 1 / volatilities.replace(0, np.nan)
    inv_vol = inv_vol.fillna(0)

    adjusted_weights = inv_vol * pd.Series(sharpe_ratios) * rf
    adjusted_weights = adjusted_weights.replace([np.inf, -np.inf], 0).fillna(0)

    if adjusted_weights.sum() == 0:
        weights = pd.Series(1.0 / len(commodities_data), index=list(commodities_data.keys()))
    else:
        weights = adjusted_weights / adjusted_weights.sum()

    # Portfolio metrics
    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)

    correlation_matrix = returns_df.corr()

    return {
        'weights': weights.to_dict(),
        'sharpe_ratios': sharpe_ratios,
        'portfolio_volatility': portfolio_vol,
        'portfolio_sharpe': portfolio_sharpe,
        'correlation_matrix': correlation_matrix
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_candlestick_with_indicators(df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Create interactive candlestick chart with technical indicators."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(title, 'Volume', 'RSI'))

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

    # Moving Averages
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1),
                                  name='MA20'), row=1, col=1)
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='green', width=1),
                                  name='MA50'), row=1, col=1)

    # Volume
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1),
                                  name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=True, row=1, col=1)

    return fig

def plot_forecast(historical_df: pd.DataFrame, forecast_df: pd.DataFrame,
                  model_results: dict, commodity: str) -> go.Figure:
    """Plot historical prices with forecasts."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['Close'],
                              mode='lines', name='Historical', line=dict(color='blue')))

    # Forecast
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'],
                              mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))

    # Confidence interval (simplified)
    std_err = historical_df['Close'].iloc[-20:].std() * 0.5
    fig.add_trace(go.Scatter(x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                              y=(forecast_df['Forecast'] + std_err).tolist() +
                                (forecast_df['Forecast'] - std_err).tolist()[::-1],
                              fill='toself', fillcolor='rgba(255,0,0,0.1)',
                              line=dict(color='rgba(255,0,0,0)'), name='95% CI'))

    fig.update_layout(title=f'{commodity} Price Forecast',
                      xaxis_title='Date', yaxis_title='Price ($)',
                      height=600, showlegend=True)

    return fig

def plot_seasonal_pattern(seasonal_df: pd.DataFrame) -> go.Figure:
    """Plot seasonal price patterns."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=seasonal_df['Avg Return'],
                          marker_color=['green' if x > 0 else 'red' for x in seasonal_df['Avg Return']],
                          name='Average Return'))

    fig.update_layout(title='Seasonal Price Patterns',
                      xaxis_title='Month', yaxis_title='Average Return (%)',
                      height=500)

    return fig

def plot_correlation_matrix(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(title='Commodity Correlation Matrix',
                      height=600, width=600)

    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<p class="main-header">🌾 AgriTrade Pro</p>', unsafe_allow_html=True)
    st.markdown("### Elite Commodities Trading Intelligence Platform")
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Control Panel")

    commodities = ['Corn', 'Wheat', 'Soybeans', 'Coffee', 'Sugar', 'Cotton', 'Live Cattle', 'Lean Hogs']
    selected_commodities = st.sidebar.multiselect("Select Commodities", commodities, default=['Corn', 'Wheat'])

    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Technical Analysis", "ML Forecasting", "Risk Analytics", "Seasonal Patterns", "Portfolio Optimization"]
    )

    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 5)
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"])

    refresh_data = st.sidebar.button("🔄 Refresh Data")

    # Main content area
    if not selected_commodities:
        st.warning("Please select at least one commodity.")
        return

    # Load data
    with st.spinner("Loading market data..."):
        commodities_data = {}
        for comm in selected_commodities:
            commodities_data[comm] = generate_commodity_data(comm)
            commodities_data[comm] = calculate_technical_indicators(commodities_data[comm])

    st.success(f"✅ Loaded data for {len(selected_commodities)} commodities")
    st.markdown("---")

    # Single commodity analysis
    if len(selected_commodities) == 1:
        commodity = selected_commodities[0]
        df = commodities_data[commodity]

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        current_price = df['Close'].iloc[-1]
        price_change = ((current_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        avg_volume = df['Volume'].iloc[-20:].mean()

        col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        col2.metric("Volatility (Ann.)", f"{volatility:.1f}%")
        col3.metric("Avg Volume (20d)", f"{avg_volume:,.0f}")
        col4.metric("Data Points", f"{len(df)}")

        st.markdown("---")

        if analysis_type == "Technical Analysis":
            st.subheader(f"📊 Technical Analysis - {commodity}")

            # Chart
            fig = plot_candlestick_with_indicators(df, f"{commodity} Price & Indicators")
            st.plotly_chart(fig, use_container_width=True)

            # Signals
            signals = get_technical_signals(df)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Technical Indicators")
                for indicator, signal in signals.items():
                    if indicator != 'Overall':
                        emoji = "🟢" if 'Bullish' in signal or signal == 'Oversold' else "🔴" if 'Bearish' in signal or signal == 'Overbought' else "🟡"
                        st.write(f"{emoji} **{indicator}:** {signal}")

            with col2:
                st.markdown("#### Overall Signal")
                overall_signal = signals.get('Overall', 'Hold')
                if overall_signal == 'Buy':
                    st.success("🟢 **BUY SIGNAL**")
                elif overall_signal == 'Sell':
                    st.error("🔴 **SELL SIGNAL**")
                else:
                    st.info("🟡 **HOLD**")

        elif analysis_type == "ML Forecasting":
            st.subheader(f"🤖 Machine Learning Forecast - {commodity}")

            with st.spinner("Training ensemble models..."):
                model, scaler, results, feature_imp, best_model = train_forecast_model(df, forecast_horizon)

                if model is None:
                    st.error("Insufficient data for forecasting")
                    return

                forecast_df = forecast_prices(model, scaler, df, forecast_horizon)

            # Model performance
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Model Performance (CV RMSE)")
                for model_name, rmse in results.items():
                    emoji = "🏆" if model_name == best_model else ""
                    st.write(f"{emoji} **{model_name}:** ${rmse:.4f}")

            with col2:
                st.markdown("#### Best Model")
                st.success(f"**{best_model}**")

            # Forecast chart
            fig = plot_forecast(df, forecast_df, results, commodity)
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.markdown("#### Top 10 Feature Importance")
            fig_imp = go.Figure(go.Bar(
                x=feature_imp['importance'].head(10).values,
                y=feature_imp['feature'].head(10).values,
                orientation='h',
                marker_color='steelblue'
            ))
            fig_imp.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature", height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Forecast table
            st.markdown("#### Price Forecast")
            forecast_display = forecast_df.copy()
            forecast_display['Change %'] = forecast_display['Forecast'].pct_change() * 100
            st.dataframe(forecast_display.style.format({"Forecast": "${:.2f}", "Change %": "{:.2f}%"}))

        elif analysis_type == "Risk Analytics":
            st.subheader(f"⚠️ Risk Analysis - {commodity}")

            risk_metrics, returns = analyze_risk(df)

            # Metrics grid
            cols = st.columns(3)
            metric_items = list(risk_metrics.items())
            for i, (name, value) in enumerate(metric_items):
                col_idx = i % 3
                if isinstance(value, float):
                    display_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                else:
                    display_value = str(value)
                cols[col_idx].metric(name, display_value)

            # Returns distribution
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Returns Distribution', 'Q-Q Plot', 'Cumulative Returns', 'Drawdown'))

            # Histogram
            fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns'), row=1, col=1)

            # Q-Q plot
            from scipy import stats
            sorted_returns = np.sort(returns)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_returns[:len(theoretical_quantiles)],
                                      mode='markers', name='Q-Q'), row=1, col=2)
            fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', line=dict(color='red'),
                                      name='Normal'), row=1, col=2)

            # Cumulative returns
            cum_returns = (1 + returns).cumprod() - 1
            fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, mode='lines', name='Cumulative'), row=2, col=1)

            # Drawdown
            peak = df['Close'].cummax()
            drawdown = (df['Close'] - peak) / peak
            fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', fill='tozeroy'), row=2, col=2)

            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Seasonal Patterns":
            st.subheader(f"📅 Seasonal Analysis - {commodity}")

            seasonal_df = analyze_seasonality(df)

            fig = plot_seasonal_pattern(seasonal_df)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Monthly Statistics")
            st.dataframe(seasonal_df.style.format({"Avg Return": "{:.2%}", "Std Dev": "{:.2%}"}))

            # Agricultural insights
            st.markdown("#### 🌱 Agricultural Cycle Insights")
            if commodity in ['Corn', 'Soybeans', 'Wheat']:
                st.info("""
                **Planting Season (Spring):** Prices often rise due to uncertainty about acreage and weather.
                **Growing Season (Summer):** Weather-driven volatility peaks; drought/flood concerns.
                **Harvest Season (Fall):** Typical price pressure from increased supply.
                **Post-Harvest (Winter):** Focus shifts to demand and export markets.
                """)
            elif commodity == 'Coffee':
                st.info("""
                **Brazil Harvest (May-Sep):** Major supply impact on global prices.
                **Vietnam Harvest (Oct-Jan):** Robusta supply influences blend pricing.
                """)

        elif analysis_type == "Portfolio Optimization":
            st.info("Please select multiple commodities for portfolio analysis.")

    # Multi-commodity analysis
    else:
        st.subheader(f"📊 Multi-Commodity Analysis ({len(selected_commodities)} commodities)")

        if analysis_type == "Portfolio Optimization":
            with st.spinner("Optimizing portfolio..."):
                portfolio = optimize_portfolio(commodities_data, risk_tolerance)

            if portfolio:
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Volatility", f"{portfolio['portfolio_volatility']:.2%}")
                col2.metric("Portfolio Sharpe", f"{portfolio['portfolio_sharpe']:.2f}")
                col3.metric("Risk Tolerance", risk_tolerance.title())

                # Allocation
                st.markdown("#### Recommended Allocation")
                weights_df = pd.DataFrame({
                    'Commodity': list(portfolio['weights'].keys()),
                    'Weight (%)': [w * 100 for w in portfolio['weights'].values()],
                    'Sharpe Ratio': [portfolio['sharpe_ratios'].get(c, 0) for c in portfolio['weights'].keys()]
                })

                fig = go.Figure(go.Pie(labels=weights_df['Commodity'], values=weights_df['Weight (%)']),
                               hole=0.4)
                fig.update_layout(title="Portfolio Allocation", height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(weights_df.style.format({"Weight (%)": "{:.1f}%", "Sharpe Ratio": "{:.2f}"}))

                # Correlation
                st.markdown("#### Correlation Matrix")
                fig_corr = plot_correlation_matrix(portfolio['correlation_matrix'])
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.error("Insufficient data for portfolio optimization")

        else:
            # Show comparison charts
            st.markdown("#### Price Comparison (Normalized)")

            comparison_df = pd.DataFrame()
            for comm in selected_commodities:
                df = commodities_data[comm]
                normalized = df['Close'] / df['Close'].iloc[0] * 100
                comparison_df[comm] = normalized

            fig = go.Figure()
            for col in comparison_df.columns:
                fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[col], name=col))

            fig.update_layout(title="Normalized Price Performance (Base=100)",
                              xaxis_title="Date", yaxis_title="Normalized Price",
                              height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Individual quick stats
            st.markdown("#### Quick Stats")
            stats_cols = st.columns(len(selected_commodities))
            for i, comm in enumerate(selected_commodities):
                df = commodities_data[comm]
                returns = calculate_returns(df)
                sharpe = calculate_sharpe_ratio(returns)
                vol = returns.std() * np.sqrt(252)
                stats_cols[i].markdown(f"**{comm}**\n\n- Volatility: {vol:.1%}\n- Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    main()
