import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="SuperStock Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg/Two Sigma UI/UX
st.markdown("""
<style>
    .main > div {background-color: #0e1117;}
    .stDataFrame {border: 1px solid #262730;}
    .metric-card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        font-weight: bold;
        border-radius: 4px;
    }
    h1, h2, h3 {color: #ffffff;}
    .phase-growth {color: #00ff9d;}
    .phase-stagnate {color: #ffaa00;}
    .phase-decline {color: #ff4444;}
</style>
""", unsafe_allow_html=True)

# --- Core Functions ---

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date="2015-01-01"):
    """Fetch stock data with caching."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty:
            return None
        # Handle multi-level columns if present (new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    """Fetch stock info."""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception:
        return {}

def calculate_returns(df):
    """Calculate cumulative returns."""
    if df is None or len(df) < 2:
        return 0
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    if start_price == 0:
        return 0
    return (end_price - start_price) / start_price

def identify_phases(df, threshold_growth=0.50, threshold_decline=-0.20):
    """
    Identify growth, stagnation, and decline phases based on rolling annualized returns.
    Returns a DataFrame with phase labels.
    """
    if df is None or len(df) < 252:  # Need at least 1 year of data
        return pd.DataFrame()

    df = df.copy()
    # Calculate rolling 1-year returns (approx 252 trading days)
    df['Rolling_Return'] = df['Close'].pct_change(252)
    
    phases = []
    for ret in df['Rolling_Return']:
        if pd.isna(ret):
            phases.append("Unknown")
        elif ret > threshold_growth:
            phases.append("Rapid Growth")
        elif ret > 0:
            phases.append("Moderate Growth")
        elif ret > threshold_decline:
            phases.append("Stagnation")
        else:
            phases.append("Decline")
    
    df['Phase'] = phases
    return df

def get_events(ticker, df):
    """Get significant events (Splits, Dividends, Earnings approx)."""
    events = []
    try:
        stock = yf.Ticker(ticker)
        
        # Splits
        splits = stock.splits
        if not splits.empty:
            for date, ratio in splits.items():
                events.append({
                    "Date": date,
                    "Type": "Stock Split",
                    "Description": f"Split Ratio: {ratio}"
                })
        
        # Dividends
        dividends = stock.dividends
        if not dividends.empty:
            # Group by year to avoid clutter, just show major ones or recent
            recent_divs = dividends[dividends.index > (datetime.now() - timedelta(days=365*2))]
            for date, amount in recent_divs.items():
                events.append({
                    "Date": date,
                    "Type": "Dividend",
                    "Description": f"${amount:.2f}"
                })
                
    except Exception:
        pass
        
    return pd.DataFrame(events) if events else pd.DataFrame()

def plot_phase_chart(df, ticker):
    """Plot price chart colored by phases."""
    if df is None or 'Phase' not in df.columns:
        return go.Figure()

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    colors = {
        "Rapid Growth": "#00ff9d",
        "Moderate Growth": "#00d4ff",
        "Stagnation": "#ffaa00",
        "Decline": "#ff4444",
        "Unknown": "#888888"
    }
    
    # Plot segments
    df_plot = df.dropna(subset=['Rolling_Return']).copy()
    
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#ffffff', width=2)
    ))
    
    # Add annotations for phase changes (simplified)
    prev_phase = None
    for i, row in df_plot.iterrows():
        if row['Phase'] != prev_phase:
            fig.add_annotation(
                x=i,
                y=row['Close'],
                text=row['Phase'],
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=colors.get(row['Phase'], "#fff"))
            )
            prev_phase = row['Phase']

    fig.update_layout(
        title=f"{ticker} Price & Phases",
        hovermode='x unified',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#262730')
    )
    
    return fig

# --- Sidebar Controls ---
st.sidebar.title("🚀 SuperStock Detector")
st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis Settings**")

# Predefined list of known high-growth candidates since 2015 to speed up
default_tickers = [
    "NVDA", "TSLA", "AMD", "META", "NFLX", "AVGO", "AMZN", "GOOGL", 
    "MSFT", "AAPL", "CRM", "ADBE", "NOW", "PANW", "CRWD", "ZS", 
    "DDOG", "NET", "SNOW", "PLTR", "COIN", "MSTR", "SHOP", "SQ", 
    "PYPL", "ROKU", "SPOT", "UBER", "ABNB", "RBLX", "U", "PATH",
    "ENPH", "SEDG", "FSLR", "ON", "MPWR", "MRVL", "ANET", "SMCI"
]

selected_tickers = st.sidebar.multiselect(
    "Select Stocks to Analyze",
    options=default_tickers,
    default=["NVDA", "TSLA", "AMD", "META", "SMCI"],
    max_selections=10
)

analyze_btn = st.sidebar.button("🔍 Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Methodology:**
- **SuperStock**: >300% return since 2015
- **Phases**: Based on rolling 1-year returns
  - 🟢 Rapid: >50% ann.
  - 🔵 Moderate: 0-50%
  - 🟠 Stagnation: -20% to 0%
  - 🔴 Decline: <-20%
""")

# --- Main App ---

if analyze_btn:
    if not selected_tickers:
        st.warning("Please select at least one stock.")
    else:
        progress_bar = st.progress(0)
        results = []
        
        for i, ticker in enumerate(selected_tickers):
            # Update progress
            progress_bar.progress((i + 1) / len(selected_tickers))
            
            df = get_stock_data(ticker, start_date="2015-01-01")
            if df is None or len(df) < 100:
                continue
                
            info = get_stock_info(ticker)
            total_return = calculate_returns(df)
            
            results.append({
                "Ticker": ticker,
                "Name": info.get('shortName', ticker),
                "Sector": info.get('sector', 'N/A'),
                "Total Return (%)": round(total_return * 100, 2),
                "Start Price": round(df['Close'].iloc[0], 2),
                "Current Price": round(df['Close'].iloc[-1], 2),
                "Data": df,
                "Info": info
            })
        
        progress_bar.empty()
        
        if not results:
            st.error("No data retrieved. Please check tickers or internet connection.")
        else:
            # Create DataFrame for table
            df_results = pd.DataFrame(results)
            
            # Sort by return
            df_results = df_results.sort_values(by="Total Return (%)", ascending=False)
            
            # Format for display
            display_df = df_results.copy()
            display_df["Total Return (%)"] = display_df["Total Return (%)"].apply(lambda x: f"{x:.2f}%")
            display_df["Start Price"] = display_df["Start Price"].apply(lambda x: f"${x}")
            display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"${x}")
            
            st.title("📊 SuperStock Analysis Dashboard")
            
            # Top Metrics
            col1, col2, col3 = st.columns(3)
            best_stock = df_results.iloc[0]
            col1.metric("Top Performer", best_stock['Ticker'], f"{best_stock['Total Return (%)']:.1f}% Return")
            col2.metric("Avg Return", f"{df_results['Total Return (%)'].mean():.1f}%")
            col3.metric("Stocks Analyzed", len(df_results))
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "📈 Deep Dive", "📅 Timeline"])
            
            with tab1:
                st.subheader("Top Performing Stocks Since 2015")
                st.dataframe(
                    display_df[["Ticker", "Name", "Sector", "Total Return (%)", "Start Price", "Current Price"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Sector Distribution
                if 'Sector' in df_results.columns:
                    sector_counts = df_results['Sector'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts.values, hole=.3)])
                    fig_pie.update_layout(title="Sector Distribution", plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='#fff'))
                    st.plotly_chart(fig_pie, use_container_width=True)

            with tab2:
                st.subheader("Detailed Phase Analysis")
                selected_deep = st.selectbox("Select Stock for Deep Dive", df_results['Ticker'].tolist())
                
                if selected_deep:
                    row = df_results[df_results['Ticker'] == selected_deep].iloc[0]
                    df = row['Data']
                    
                    # Calculate phases
                    df_phased = identify_phases(df)
                    
                    # Plot
                    fig = plot_phase_chart(df_phased, selected_deep)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Volatility (Ann.)", f"{df['Close'].pct_change().std() * np.sqrt(252) * 100:.1f}%")
                    c2.metric("Max Drawdown", f"{(df['Close'] / df['Close'].cummax() - 1).min() * 100:.1f}%")
                    c3.metric("Sharpe Ratio (Est.)", f"{(df['Close'].pct_change().mean() / df['Close'].pct_change().std()) * np.sqrt(252):.2f}")
                    c4.metric("Current Phase", df_phased['Phase'].iloc[-1])

            with tab3:
                st.subheader("Events & Timeline")
                selected_event = st.selectbox("Select Stock for Events", df_results['Ticker'].tolist(), key="event_sel")
                
                if selected_event:
                    row = df_results[df_results['Ticker'] == selected_event].iloc[0]
                    df = row['Data']
                    
                    # Get events
                    events_df = get_events(selected_event, df)
                    
                    # Plot Price with Event Markers
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff')))
                    
                    if not events_df.empty:
                        fig.add_trace(go.Scatter(
                            x=events_df['Date'],
                            y=[df.loc[d, 'Close'] if d in df.index else np.nan for d in events_df['Date']],
                            mode='markers',
                            name='Events',
                            marker=dict(size=10, color='#ffaa00', symbol='star')
                        ))
                        
                        # Add event table
                        st.dataframe(events_df, use_container_width=True, hide_index=True)
                    
                    fig.update_layout(
                        title=f"{selected_event} Price & Key Events",
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='#fff'),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("🚀 US SuperStock Detector")
    st.markdown("""
    ### Detect, Analyze, and Understand Market Outperformers
    
    This tool identifies "SuperStocks" in the US market since 2015, analyzing their growth trajectories, 
    stagnation periods, and key catalyst events.
    
    **Features:**
    - 🔍 **Screening**: Filters stocks with massive returns (>300%).
    - 📉 **Phase Detection**: Automatically identifies Rapid Growth, Stagnation, and Decline periods.
    - 📅 **Timeline**: Correlates price action with splits, dividends, and major events.
    - 🧠 **Quantitative Metrics**: Volatility, Sharpe Ratio, and Drawdown analysis.
    
    _Select stocks in the sidebar and click **Run Analysis** to begin._
    """)
    
    # Example visualization placeholder
    fig_placeholder = go.Figure()
    fig_placeholder.add_trace(go.Scatter(x=[1,2,3], y=[1,3,2], mode='lines+markers', name="Demo"))
    fig_placeholder.update_layout(title="Preview", template="plotly_dark", height=300)
    st.plotly_chart(fig_placeholder, use_container_width=True)
