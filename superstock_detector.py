import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SuperStock Detector US",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg/Two Sigma aesthetic
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #3fb950;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    /* Hide default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117; 
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #58a6ff; 
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA UTILS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date="2015-01-01"):
    """Fetch historical data with caching."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty or len(df) == 0:
            return None
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

@st.cache_data(ttl=86400)
def get_stock_info(ticker):
    """Fetch static info with caching."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception:
        return {}

def calculate_returns(df):
    """Calculate key return metrics."""
    if df is None or len(df) < 2:
        return None
    
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    total_return = (end_price - start_price) / start_price
    
    # Annualized Return
    years = (df.index[-1] - df.index[0]).days / 365.25
    if years > 0:
        cagr = (end_price / start_price) ** (1/years) - 1
    else:
        cagr = 0
        
    # Volatility
    daily_returns = df['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Max Drawdown
    rolling_max = df['Close'].cummax()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Sharpe Ratio (assuming risk-free rate ~2%)
    rf = 0.02
    sharpe = (cagr - rf) / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'start_price': start_price,
        'end_price': end_price
    }

def identify_phases(df, metrics):
    """Identify growth, stagnation, and decline phases."""
    if df is None:
        return []
    
    phases = []
    window = 60  # ~3 months
    
    closes = df['Close'].values
    dates = df.index
    
    # Simple rolling CAGR approximation
    for i in range(window, len(closes), window):
        start_p = closes[i-window]
        end_p = closes[i]
        period_ret = (end_p - start_p) / start_p
        ann_ret = (1 + period_ret) ** (252/window) - 1
        
        phase_type = "Stagnation"
        color = "#8b949e" # Gray
        
        if ann_ret > 0.50:
            phase_type = "Rapid Growth"
            color = "#3fb950" # Green
        elif ann_ret > 0.10:
            phase_type = "Moderate Growth"
            color = "#58a6ff" # Blue
        elif ann_ret < -0.10:
            phase_type = "Decline"
            color = "#f85149" # Red
            
        phases.append({
            'start': dates[i-window],
            'end': dates[i],
            'type': phase_type,
            'color': color,
            'return': ann_ret
        })
        
    return phases

def get_events(ticker, start_date, end_date):
    """Fetch earnings, splits, and dividends."""
    events = []
    try:
        stock = yf.Ticker(ticker)
        
        # Earnings
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            for date, row in earnings.iterrows():
                if start_date <= date.date() <= end_date:
                    events.append({
                        'date': date,
                        'type': 'Earnings',
                        'desc': f"EPS: {row['EPS Estimate']:.2f}" if pd.notna(row['EPS Estimate']) else "Earnings Report"
                    })
        
        # Splits
        splits = stock.splits
        if splits is not None and not splits.empty:
            for date, ratio in splits.items():
                if start_date <= date.date() <= end_date:
                    events.append({
                        'date': date,
                        'type': 'Split',
                        'desc': f"Split Ratio: {ratio}"
                    })
                    
        # Dividends
        divs = stock.dividends
        if divs is not None and not divs.empty:
            for date, amount in divs.items():
                if start_date <= date.date() <= end_date and amount > 0:
                    events.append({
                        'date': date,
                        'type': 'Dividend',
                        'desc': f"${amount:.2f}"
                    })
    except Exception:
        pass
        
    return sorted(events, key=lambda x: x['date'])

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

def main():
    # Sidebar Controls
    st.sidebar.title("🎛️ Control Panel")
    
    # Pre-defined list of known superstocks candidates to ensure speed
    # Focused on high-beta, high-growth tech/healthcare since 2015
    default_tickers = [
        "NVDA", "TSLA", "AMD", "META", "NFLX", "AMZN", "GOOGL", "MSFT", 
        "AVGO", "ASML", "SMCI", "PLTR", "COIN", "MARA", "RIOT", "ENPH", 
        "SEDG", "MRNA", "BNTX", "ZM", "DOCU", "CRWD", "SNOW", "NET", 
        "DDOG", "ESTC", "SHOP", "SQ", "PYPL", "ROKU", "SPOT", "UBER", 
        "ABNB", "LYFT", "RBLX", "U", "PATH", "AI", "IONQ", "RGTI"
    ]
    
    selected_tickers = st.sidebar.multiselect(
        "Select Stocks to Analyze",
        options=default_tickers,
        default=["NVDA", "TSLA", "AMD", "META", "SMCI"],
        max_selections=10
    )
    
    analyze_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.title("🚀 US SuperStock Detector (Since 2015)")
    st.markdown("""
    **Objective**: Identify stocks with exponential growth trajectories, pinpoint stagnation periods, 
    and correlate price action with fundamental events (Earnings, Splits, News).
    """)
    
    if not selected_tickers:
        st.warning("Please select at least one stock from the sidebar.")
        return

    if analyze_btn or 'data_ready' in st.session_state:
        st.session_state.data_ready = True
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        all_events = {}
        
        # Parallel Processing for Speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(analyze_single_stock, t): t for t in selected_tickers}
            
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data, events = future.result()
                    if data:
                        results.append(data)
                        all_events[ticker] = events
                    completed += 1
                    progress_bar.progress(completed / len(selected_tickers))
                    status_text.text(f"Analyzing: {ticker} ({completed}/{len(selected_tickers)})")
                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {e}")
        
        status_text.text("Analysis Complete!")
        
        if not results:
            st.error("No valid data retrieved. Check your internet connection or ticker symbols.")
            return
            
        # Create DataFrame for Overview
        df_results = pd.DataFrame(results)
        
        # Sort by Total Return
        df_results = df_results.sort_values(by='total_return_pct', ascending=False)
        
        # Display Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Deep Dive", "📅 Event Timeline", "📉 Quantitative Metrics"])
        
        with tab1:
            render_overview(df_results)
            
        with tab2:
            render_deep_dive(df_results, all_events)
            
        with tab3:
            render_timeline(df_results, all_events)
            
        with tab4:
            render_quant_metrics(df_results)

def analyze_single_stock(ticker):
    """Worker function for parallel analysis."""
    df = get_stock_data(ticker, start_date="2015-01-01")
    if df is None or len(df) < 60:
        return None, []
        
    metrics = calculate_returns(df)
    if not metrics:
        return None, []
        
    info = get_stock_info(ticker)
    sector = info.get('sector', 'Unknown')
    industry = info.get('industry', 'Unknown')
    market_cap = info.get('marketCap', 0)
    
    phases = identify_phases(df, metrics)
    events = get_events(ticker, df.index[0].date(), datetime.now().date())
    
    return {
        'ticker': ticker,
        'sector': sector,
        'industry': industry,
        'market_cap': market_cap,
        'current_price': metrics['end_price'],
        'start_price': metrics['start_price'],
        'total_return_pct': metrics['total_return'] * 100,
        'cagr_pct': metrics['cagr'] * 100,
        'volatility_pct': metrics['volatility'] * 100,
        'max_drawdown_pct': metrics['max_drawdown'] * 100,
        'sharpe_ratio': metrics['sharpe'],
        'phases': phases,
        'df': df
    }, events

def render_overview(df):
    st.subheader("Top Performing SuperStocks")
    
    # Format for display
    display_df = df.copy()
    
    # Formatters
    def fmt_pct(x):
        return f"{x:.2f}%"
    def fmt_float(x):
        return f"{x:.2f}"
    def fmt_curr(x):
        return f"${x:,.2f}"
        
    # Select columns
    cols = ['ticker', 'sector', 'current_price', 'total_return_pct', 'cagr_pct', 'sharpe_ratio', 'max_drawdown_pct']
    show_df = display_df[cols].copy()
    
    # Color mapping for returns
    def color_returns(val):
        if val > 100:
            color = '#3fb950' # Bright Green
        elif val > 0:
            color = '#58a6ff' # Blue
        else:
            color = '#f85149' # Red
        return f'color: {color}; font-weight: bold'

    # Apply styling
    styled_df = show_df.style.format({
        'current_price': '${:,.2f}',
        'total_return_pct': '{:.2f}%',
        'cagr_pct': '{:.2f}%',
        'max_drawdown_pct': '{:.2f}%',
        'sharpe_ratio': '{:.2f}'
    }).applymap(color_returns, subset=['total_return_pct'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Top Chart
    top_5 = df.nlargest(5, 'total_return_pct')
    fig = px.bar(
        top_5, x='ticker', y='total_return_pct', 
        title="Top 5 Returns Since 2015 (%)",
        color='total_return_pct',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#c9d1d9')
    st.plotly_chart(fig, use_container_width=True)

def render_deep_dive(df, events_dict):
    st.subheader("Individual Stock Analysis")
    
    selected = st.selectbox("Choose a stock for deep dive", options=df['ticker'].tolist())
    
    if not selected:
        return
        
    row = df[df['ticker'] == selected].iloc[0]
    full_df = row['df']
    phases = row['phases']
    
    # Price Chart with Phases
    fig = go.Figure()
    
    # Main Price Line
    fig.add_trace(go.Scatter(
        x=full_df.index, y=full_df['Close'],
        mode='lines', name='Price',
        line=dict(color='#58a6ff', width=2)
    ))
    
    # Add Phase Backgrounds (Simplified as shapes)
    # Note: Plotly shapes can be heavy, limiting to major phases
    for i, phase in enumerate(phases):
        if phase['type'] == 'Rapid Growth':
            fig.add_vrect(
                x0=phase['start'], x1=phase['end'],
                fillcolor="#3fb950", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Growth", annotation_position="top left"
            )
        elif phase['type'] == 'Decline':
            fig.add_vrect(
                x0=phase['start'], x1=phase['end'],
                fillcolor="#f85149", opacity=0.1,
                layer="below", line_width=0
            )
            
    fig.update_layout(
        title=f"{selected} Price Action & Regimes",
        hovermode='x unified',
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font_color='#c9d1d9',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stagnation Analysis
    stagnation_periods = [p for p in phases if p['type'] == 'Stagnation']
    if stagnation_periods:
        st.markdown("### 🛑 Identified Stagnation Periods")
        stag_df = pd.DataFrame(stagnation_periods)
        stag_df['duration_days'] = (stag_df['end'] - stag_df['start']).dt.days
        st.dataframe(stag_df[['start', 'end', 'duration_days']], use_container_width=True, hide_index=True)
    else:
        st.success("No significant stagnation periods detected (consistent growth or volatility).")

def render_timeline(df, events_dict):
    st.subheader("Corporate Events Timeline")
    
    selected = st.selectbox("Select Stock for Timeline", options=df['ticker'].tolist(), key="timeline_sel")
    
    if selected and selected in events_dict:
        events = events_dict[selected]
        if not events:
            st.info("No specific events (splits/dividends/earnings) found in cache for this period.")
        else:
            # Create event dataframe
            ev_df = pd.DataFrame(events)
            
            # Plot
            fig = go.Figure()
            
            # Price
            price_df = df[df['ticker'] == selected]['df'].iloc[0]
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df['Close'],
                mode='lines', name='Price',
                line=dict(color='#8b949e', width=1)
            ))
            
            # Event Markers
            for type_name, color in zip(['Earnings', 'Split', 'Dividend'], ['#f85149', '#58a6ff', '#3fb950']):
                type_events = ev_df[ev_df['type'] == type_name]
                if not type_events.empty:
                    # Map events to price roughly
                    y_positions = []
                    for _, row in type_events.iterrows():
                        try:
                            idx = price_df.index.get_loc(row['date'], method='nearest')
                            y_positions.append(price_df['Close'].iloc[idx])
                        except:
                            y_positions.append(price_df['Close'].mean())
                            
                    fig.add_trace(go.Scatter(
                        x=type_events['date'],
                        y=y_positions,
                        mode='markers+text',
                        name=type_name,
                        marker=dict(color=color, size=10, symbol='triangle-up'),
                        text=type_events['desc'],
                        textposition="top center"
                    ))
            
            fig.update_layout(
                title=f"Events Overlay: {selected}",
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font_color='#c9d1d9',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

def render_quant_metrics(df):
    st.subheader("Quantitative Risk & Performance Matrix")
    
    metrics_df = df[['ticker', 'cagr_pct', 'volatility_pct', 'sharpe_ratio', 'max_drawdown_pct']].copy()
    
    # Heatmap of correlation
    corr_matrix = metrics_df[['cagr_pct', 'volatility_pct', 'sharpe_ratio', 'max_drawdown_pct']].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Metric Correlations"
    )
    fig.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#c9d1d9')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Stats Table
    st.dataframe(metrics_df.style.format("{:.2f}").background_gradient(cmap='viridis', subset=['sharpe_ratio']), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
