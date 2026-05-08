"""
MDB Project & Loan Portfolio Monitor
Streamlit Interactive Dashboard for AIIB Digital Program Associate – Data Analysis Role

This dashboard simulates a multilateral development bank's project and loan data ecosystem,
demonstrating working knowledge of project and loan products.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO

# Import data generator
from data_generator import generate_all_data

# Page configuration
st.set_page_config(
    page_title="MDB Project & Loan Portfolio Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Primary colors */
    :root {
        --primary-color: #1a3a5c;
        --accent-color: #d4a843;
        --bg-color: #f5f5f5;
    }
    
    /* Main container */
    .main > div {
        background-color: var(--bg-color);
    }
    
    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--accent-color);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--primary-color);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--primary-color);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Table styling */
    table {
        font-size: 14px;
    }
    
    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, #2a4a6c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def format_currency(value):
    """Format value as USD currency."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def format_percentage(value):
    """Format value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


@st.cache_data
def load_data():
    """Load and cache the generated data."""
    return generate_all_data()


def render_project_overview_tab(projects_df, loans_df):
    """Render the Project Overview tab."""
    
    st.header("📊 Project Overview")
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_portfolio = projects_df["Total Project Cost (USD)"].sum()
    active_projects = len(projects_df[projects_df["Status"].isin(["Implementation", "Appraisal"])])
    total_disbursed = loans_df["Disbursed Amount (USD)"].sum()
    total_commitment = loans_df["Commitment Amount (USD)"].sum()
    disbursement_ratio = (total_disbursed / total_commitment * 100) if total_commitment > 0 else 0
    
    with col1:
        st.metric(
            label="Total Portfolio Value",
            value=f"${total_portfolio/1e9:.2f}B",
            delta=None
        )
    with col2:
        st.metric(
            label="Active Projects",
            value=active_projects,
            delta=None
        )
    with col3:
        st.metric(
            label="Total Disbursed",
            value=f"${total_disbursed/1e9:.2f}B",
            delta=None
        )
    with col4:
        st.metric(
            label="Disbursement Ratio",
            value=f"{disbursement_ratio:.1f}%",
            delta=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("🔍 Filter Projects")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        sector_filter = st.multiselect(
            "Sector",
            options=projects_df["Sector"].unique(),
            default=projects_df["Sector"].unique()
        )
    
    with filter_col2:
        country_filter = st.multiselect(
            "Country",
            options=projects_df["Country"].unique(),
            default=projects_df["Country"].unique()
        )
    
    with filter_col3:
        status_filter = st.multiselect(
            "Status",
            options=projects_df["Status"].unique(),
            default=projects_df["Status"].unique()
        )
    
    # Apply filters
    filtered_projects = projects_df[
        (projects_df["Sector"].isin(sector_filter)) &
        (projects_df["Country"].isin(country_filter)) &
        (projects_df["Status"].isin(status_filter))
    ]
    
    st.markdown(f"**{len(filtered_projects)} projects** matching selected filters")
    
    # Project table
    st.subheader("📋 Project Portfolio")
    
    display_cols = ["Project ID", "Project Name", "Country", "Sector", "Status",
                   "Total Project Cost (USD)", "AIIB Loan Amount (USD)", 
                   "Co-financing Amount (USD)", "Approval Date"]
    
    formatted_df = filtered_projects[display_cols].copy()
    formatted_df["Total Project Cost (USD)"] = formatted_df["Total Project Cost (USD)"].apply(format_currency)
    formatted_df["AIIB Loan Amount (USD)"] = formatted_df["AIIB Loan Amount (USD)"].apply(format_currency)
    formatted_df["Co-financing Amount (USD)"] = formatted_df["Co-financing Amount (USD)"].apply(format_currency)
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    
    # Charts section
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Projects by Sector
        sector_counts = filtered_projects.groupby("Sector").size().reset_index(name='Count')
        fig_sector = px.pie(sector_counts, values='Count', names='Sector', 
                           title="Projects by Sector",
                           color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with chart_col2:
        # Projects by Country
        country_counts = filtered_projects.groupby("Country").size().reset_index(name='Count')
        fig_country = px.bar(country_counts, x='Country', y='Count',
                            title="Projects by Country",
                            color='Count',
                            color_continuous_scale=px.colors.sequential.Gold)
        st.plotly_chart(fig_country, use_container_width=True)
    
    # Status distribution
    st.subheader("📈 Project Status Distribution")
    status_counts = filtered_projects.groupby("Status").size().reset_index(name='Count')
    fig_status = px.bar(status_counts, x='Status', y='Count',
                       title="Projects by Status",
                       color='Status',
                       color_discrete_map={
                           "Concept": "#95a5a6",
                           "Appraisal": "#3498db",
                           "Implementation": "#f39c12",
                           "Completion": "#27ae60"
                       })
    st.plotly_chart(fig_status, use_container_width=True)


def render_loan_portfolio_tab(loans_df, projects_df):
    """Render the Loan Portfolio tab."""
    
    st.header("💰 Loan Portfolio")
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_commitment = loans_df["Commitment Amount (USD)"].sum()
    total_disbursed = loans_df["Disbursed Amount (USD)"].sum()
    total_outstanding = loans_df["Outstanding Balance (USD)"].sum()
    avg_interest = loans_df["Interest Rate (%)"].mean()
    
    with col1:
        st.metric(
            label="Total Commitments",
            value=f"${total_commitment/1e9:.2f}B",
            delta=None
        )
    with col2:
        st.metric(
            label="Total Disbursed",
            value=f"${total_disbursed/1e9:.2f}B",
            delta=None
        )
    with col3:
        st.metric(
            label="Outstanding Balance",
            value=f"${total_outstanding/1e9:.2f}B",
            delta=None
        )
    with col4:
        st.metric(
            label="Avg Interest Rate",
            value=f"{avg_interest:.2f}%",
            delta=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Classification breakdown
    st.subheader("📊 Portfolio Classification")
    
    class_col1, class_col2, class_col3 = st.columns(3)
    
    with class_col1:
        # By Loan Type
        loan_type_dist = loans_df.groupby("Loan Type")["Commitment Amount (USD)"].sum().reset_index()
        fig_loan_type = px.pie(loan_type_dist, values='Commitment Amount (USD)', names='Loan Type',
                              title="By Loan Type")
        st.plotly_chart(fig_loan_type, use_container_width=True)
    
    with class_col2:
        # By Risk Rating
        risk_dist = loans_df.groupby("Risk Rating").size().reset_index(name='Count')
        # Group ratings for better visualization
        risk_dist['Rating Category'] = risk_dist['Risk Rating'].apply(
            lambda x: 'AA+' if x in ['AAA', 'AA+', 'AA', 'AA-'] else
                      'A-BBB' if x in ['A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-'] else
                      'BB-B' if x in ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-'] else 'CCC+'
        )
        risk_cat_dist = risk_dist.groupby('Rating Category')['Count'].sum().reset_index()
        fig_risk = px.bar(risk_cat_dist, x='Rating Category', y='Count',
                         title="By Risk Category",
                         color='Count',
                         color_continuous_scale=px.colors.sequential.RdYlGn_r)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with class_col3:
        # By Covenant Status
        covenant_dist = loans_df.groupby("Covenant Status").size().reset_index(name='Count')
        fig_covenant = px.pie(covenant_dist, values='Count', names='Covenant Status',
                             title="By Covenant Status",
                             color='Covenant Status',
                             color_discrete_map={
                                 "Compliant": "#27ae60",
                                 "Watch": "#f39c12",
                                 "Breach": "#e74c3c"
                             })
        st.plotly_chart(fig_covenant, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Portfolio Concentration Analysis
    st.subheader("🎯 Portfolio Concentration Analysis")
    
    conc_col1, conc_col2 = st.columns(2)
    
    with conc_col1:
        # By Country
        country_conc = loans_df.merge(projects_df[['Project ID', 'Country']], on='Project ID')
        country_exposure = country_conc.groupby('Country')['Commitment Amount (USD)'].sum().reset_index()
        fig_country = px.treemap(country_exposure, path=['Country'], values='Commitment Amount (USD)',
                                title="Exposure by Country")
        st.plotly_chart(fig_country, use_container_width=True)
    
    with conc_col2:
        # By Sector
        sector_conc = loans_df.merge(projects_df[['Project ID', 'Sector']], on='Project ID')
        sector_exposure = sector_conc.groupby('Sector')['Commitment Amount (USD)'].sum().reset_index()
        fig_sector = px.treemap(sector_exposure, path=['Sector'], values='Commitment Amount (USD)',
                               title="Exposure by Sector")
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # NPL Tracker
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚠️ Non-Performing Loan Tracker")
    
    npl_loans = loans_df[
        (loans_df["Covenant Status"] == "Breach") | 
        (loans_df["Risk Rating"].isin(["CCC+", "CCC", "CCC-", "D"]))
    ].copy()
    
    if len(npl_loans) > 0:
        npl_display = npl_loans.merge(
            projects_df[['Project ID', 'Project Name', 'Country', 'Sector']], 
            on='Project ID'
        )
        
        npl_cols = ["Loan ID", "Project Name", "Country", "Sector", "Risk Rating", 
                   "Covenant Status", "Outstanding Balance (USD)"]
        npl_display["Outstanding Balance (USD)"] = npl_display["Outstanding Balance (USD)"].apply(format_currency)
        
        st.dataframe(npl_display[npl_cols], use_container_width=True, hide_index=True)
        
        # NPL metrics
        npl_metric_col1, npl_metric_col2 = st.columns(2)
        with npl_metric_col1:
            st.metric("Number of At-Risk Loans", len(npl_loans))
        with npl_metric_col2:
            npl_exposure = npl_loans["Outstanding Balance (USD)"].sum()
            st.metric("Total At-Risk Exposure", f"${npl_exposure/1e6:.1f}M")
    else:
        st.success("No non-performing loans detected!")
    
    # Loan Book Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Full Loan Book")
    
    loan_display = loans_df.copy()
    loan_display["Commitment Amount (USD)"] = loan_display["Commitment Amount (USD)"].apply(format_currency)
    loan_display["Disbursed Amount (USD)"] = loan_display["Disbursed Amount (USD)"].apply(format_currency)
    loan_display["Outstanding Balance (USD)"] = loan_display["Outstanding Balance (USD)"].apply(format_currency)
    
    st.dataframe(loan_display, use_container_width=True, hide_index=True)


def render_disbursement_monitoring_tab(disbursements_df, loans_df, projects_df):
    """Render the Disbursement Monitoring tab."""
    
    st.header("💸 Disbursement Monitoring")
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_scheduled = disbursements_df["Amount (USD)"].sum()
    total_disbursed = disbursements_df[disbursements_df["Status"] == "Disbursed"]["Amount (USD)"].sum()
    delayed_count = len(disbursements_df[disbursements_df["Status"] == "Delayed"])
    pending_count = len(disbursements_df[disbursements_df["Status"] == "Pending"])
    
    with col1:
        st.metric(
            label="Total Scheduled",
            value=f"${total_scheduled/1e9:.2f}B",
            delta=None
        )
    with col2:
        st.metric(
            label="Total Disbursed",
            value=f"${total_disbursed/1e9:.2f}B",
            delta=None
        )
    with col3:
        st.metric(
            label="Delayed Disbursements",
            value=delayed_count,
            delta="-" if delayed_count > 0 else None
        )
    with col4:
        st.metric(
            label="Pending Disbursements",
            value=pending_count,
            delta=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Delayed Disbursements
    st.subheader("⚠️ Delayed Disbursements")
    
    delayed_disb = disbursements_df[disbursements_df["Status"] == "Delayed"].copy()
    
    if len(delayed_disb) > 0:
        delayed_display = delayed_disb.merge(
            projects_df[['Project ID', 'Project Name']], 
            on='Project ID'
        )
        delayed_display["Delay (Days)"] = delayed_display.apply(
            lambda row: (datetime.now() - row["Scheduled Date"]).days if pd.notna(row["Scheduled Date"]) else 0,
            axis=1
        )
        
        delay_cols = ["Disbursement ID", "Project Name", "Tranche Number", 
                     "Scheduled Date", "Amount (USD)", "Delay (Days)"]
        delayed_display["Amount (USD)"] = delayed_display["Amount (USD)"].apply(format_currency)
        
        st.dataframe(delayed_display[delay_cols], use_container_width=True, hide_index=True)
        
        # Average delay
        avg_delay = delayed_display["Delay (Days)"].mean()
        st.warning(f"Average delay: **{avg_delay:.0f} days**")
    else:
        st.success("No delayed disbursements!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disbursement Progress by Project
    st.subheader("📊 Disbursement Progress by Project")
    
    progress_data = loans_df.merge(
        projects_df[['Project ID', 'Project Name']], 
        on='Project ID'
    ).head(15)  # Show top 15 projects
    
    progress_data["Progress %"] = (progress_data["Disbursed Amount (USD)"] / 
                                   progress_data["Commitment Amount (USD)"] * 100)
    
    fig_progress = px.bar(
        progress_data.sort_values("Progress %"),
        x="Progress %",
        y="Project Name",
        orientation='h',
        title="Disbursement Progress (Top 15 Projects)",
        color="Progress %",
        color_continuous_scale=px.colors.sequential.Greens
    )
    fig_progress.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_progress, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disbursement Trend Over Time
    st.subheader("📈 Disbursement Trend Over Time")
    
    disbursed_only = disbursements_df[disbursements_df["Status"] == "Disbursed"].copy()
    disbursed_only["Month"] = disbursed_only["Actual Date"].dt.to_period('M').astype(str)
    
    monthly_disb = disbursed_only.groupby("Month")["Amount (USD)"].sum().reset_index()
    monthly_disb = monthly_disb.sort_values("Month")
    
    fig_trend = px.area(
        monthly_disb,
        x="Month",
        y="Amount (USD)",
        title="Monthly Disbursement Trend",
        labels={"Month": "Month", "Amount (USD)": "Disbursed Amount (USD)"},
        color_discrete_sequence=["#1a3a5c"]
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Disbursement Schedule Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Full Disbursement Schedule")
    
    disb_display = disbursements_df.copy()
    disb_display["Amount (USD)"] = disb_display["Amount (USD)"].apply(format_currency)
    
    st.dataframe(disb_display, use_container_width=True, hide_index=True)


def render_data_quality_tab(issues_df, completeness_df, projects_df, loans_df):
    """Render the Data Quality & Reconciliation tab."""
    
    st.header("🔍 Data Quality & Reconciliation")
    st.markdown("---")
    
    st.info("""
    **About This Tab**: This section demonstrates root cause analysis capabilities for common 
    data quality issues found in project and loan management systems. It helps identify 
    discrepancies, missing data, duplicates, and classification inconsistencies.
    """)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_issues = len(issues_df)
    high_severity = len(issues_df[issues_df["Severity"] == "High"])
    unresolved = len(issues_df[issues_df["Status"] == "Unresolved"])
    avg_completeness = completeness_df["Completeness Score (%)"].mean()
    
    with col1:
        st.metric(
            label="Total Issues Detected",
            value=total_issues,
            delta=None
        )
    with col2:
        st.metric(
            label="High Severity Issues",
            value=high_severity,
            delta="-" if high_severity > 0 else None
        )
    with col3:
        st.metric(
            label="Unresolved Issues",
            value=unresolved,
            delta=None
        )
    with col4:
        st.metric(
            label="Avg Data Completeness",
            value=f"{avg_completeness:.1f}%",
            delta=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Quality Issues Table
    st.subheader("📋 Flagged Data Quality Issues")
    
    severity_color = {
        "High": "#e74c3c",
        "Medium": "#f39c12",
        "Low": "#27ae60"
    }
    
    issues_display = issues_df.copy()
    
    def color_severity(severity):
        color = severity_color.get(severity, "#95a5a6")
        return f'background-color: {color}; color: white'
    
    st.dataframe(
        issues_display.style.applymap(color_severity, subset=["Severity"]),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Completeness Scores
    st.subheader("📊 Data Completeness by Project")
    
    completeness_sorted = completeness_df.sort_values("Completeness Score (%)")
    
    fig_completeness = px.bar(
        completeness_sorted,
        x="Completeness Score (%)",
        y="Project ID",
        orientation='h',
        title="Data Completeness Score per Project",
        color="Completeness Score (%)",
        color_continuous_scale=px.colors.sequential.RdYlGn
    )
    fig_completeness.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_completeness, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reconciliation Log
    st.subheader("📝 Reconciliation Log")
    
    reconciliation_data = []
    
    for _, issue in issues_df.iterrows():
        reconciliation_data.append({
            "Issue ID": issue["Issue ID"],
            "Type": issue["Type"],
            "Severity": issue["Severity"],
            "Status": issue["Status"],
            "Action Required": "Manual review required" if issue["Status"] == "Unresolved" else "Resolved/Closed",
            "Priority": 1 if issue["Severity"] == "High" and issue["Status"] == "Unresolved" else 
                        2 if issue["Severity"] == "Medium" and issue["Status"] == "Unresolved" else 3
        })
    
    recon_df = pd.DataFrame(reconciliation_data)
    recon_df = recon_df.sort_values("Priority")
    
    st.dataframe(
        recon_df[["Issue ID", "Type", "Severity", "Status", "Action Required"]],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Issue Resolution Actions
    st.subheader("🛠️ Recommended Resolution Actions")
    
    for _, issue in issues_df[issues_df["Status"] == "Unresolved"].iterrows():
        with st.expander(f"{issue['Issue ID']}: {issue['Type']} ({issue['Severity']})"):
            st.write(f"**Description**: {issue['Description']}")
            st.write(f"**Affected Entity**: {issue['Affected Entity']}")
            
            if issue["Type"] == "Data Mismatch":
                st.write("**Recommended Action**: Reconcile loan commitment amount with project loan amount. Verify source system data.")
            elif issue["Type"] == "Missing Data":
                st.write("**Recommended Action**: Update missing fields from source documentation or contact project team.")
            elif issue["Type"] == "Duplicate Record":
                st.write("**Recommended Action**: Merge duplicate records and archive redundant entry.")
            elif issue["Type"] == "Classification Inconsistency":
                st.write("**Recommended Action**: Standardize country/sector classification using official codes.")
            elif issue["Type"] == "Data Anomaly":
                st.write("**Recommended Action**: Investigate data entry error and correct sign/value inconsistency.")


def render_bi_self_service_tab(projects_df, loans_df):
    """Render the Business Intelligence & Self-Service tab."""
    
    st.header("📈 Business Intelligence & Self-Service")
    st.markdown("---")
    
    st.info("""
    **Custom Report Builder**: Select dimensions and metrics to create custom reports. 
    Export data or visualizations for further analysis.
    """)
    
    # Dimension Selection
    st.subheader("🔧 Report Configuration")
    
    dim_col1, dim_col2 = st.columns(2)
    
    with dim_col1:
        dimensions = st.multiselect(
            "Select Dimensions",
            options=["Country", "Sector", "Loan Type", "Status", "Risk Rating", "Covenant Status"],
            default=["Country", "Sector"]
        )
    
    with dim_col2:
        metrics = st.multiselect(
            "Select Metrics",
            options=["Commitment", "Disbursed", "Outstanding", "NPL Ratio", "Project Count"],
            default=["Commitment", "Disbursed"]
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate custom report
    if dimensions and metrics:
        st.subheader("📊 Custom Report Results")
        
        # Merge data
        merged_df = loans_df.merge(
            projects_df[['Project ID', 'Country', 'Sector', 'Status']], 
            on='Project ID'
        )
        
        # Build groupby columns
        groupby_cols = []
        if "Country" in dimensions:
            groupby_cols.append("Country")
        if "Sector" in dimensions:
            groupby_cols.append("Sector")
        if "Loan Type" in dimensions:
            groupby_cols.append("Loan Type")
        if "Status" in dimensions:
            groupby_cols.append("Status")
        if "Risk Rating" in dimensions:
            groupby_cols.append("Risk Rating")
        if "Covenant Status" in dimensions:
            groupby_cols.append("Covenant Status")
        
        if groupby_cols:
            report_df = merged_df.groupby(groupby_cols).agg({
                "Commitment Amount (USD)": "sum",
                "Disbursed Amount (USD)": "sum",
                "Outstanding Balance (USD)": "sum",
                "Loan ID": "count"
            }).reset_index()
            
            report_df.columns = groupby_cols + ["Commitment", "Disbursed", "Outstanding", "Project Count"]
            
            # Calculate NPL Ratio
            npl_df = merged_df[
                (merged_df["Covenant Status"] == "Breach") | 
                (merged_df["Risk Rating"].isin(["CCC+", "CCC", "CCC-", "D"]))
            ].groupby(groupby_cols)["Outstanding Balance (USD)"].sum().reset_index()
            npl_df.columns = groupby_cols + ["NPL Amount"]
            
            report_df = report_df.merge(npl_df, on=groupby_cols, how="left")
            report_df["NPL Amount"] = report_df["NPL Amount"].fillna(0)
            report_df["NPL Ratio"] = (report_df["NPL Amount"] / report_df["Outstanding"] * 100).round(2)
            
            # Select requested metrics
            display_cols = groupby_cols.copy()
            if "Commitment" in metrics:
                display_cols.append("Commitment")
            if "Disbursed" in metrics:
                display_cols.append("Disbursed")
            if "Outstanding" in metrics:
                display_cols.append("Outstanding")
            if "NPL Ratio" in metrics:
                display_cols.append("NPL Ratio")
            if "Project Count" in metrics:
                display_cols.append("Project Count")
            
            # Format numeric columns
            for col in ["Commitment", "Disbursed", "Outstanding"]:
                if col in display_cols:
                    report_df[col] = report_df[col].apply(format_currency)
            
            st.dataframe(report_df[display_cols], use_container_width=True, hide_index=True)
            
            # Dynamic Charts
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 Visualizations")
            
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Line Chart"])
            
            if len(groupby_cols) > 0:
                x_col = groupby_cols[0]
                
                metric_for_chart = None
                if "Commitment" in metrics:
                    metric_for_chart = "Commitment"
                elif "Disbursed" in metrics:
                    metric_for_chart = "Disbursed"
                elif "Outstanding" in metrics:
                    metric_for_chart = "Outstanding"
                
                if metric_for_chart and metric_for_chart in report_df.columns:
                    # Convert back to numeric for plotting
                    plot_df = report_df.copy()
                    plot_df[metric_for_chart] = plot_df[metric_for_chart].str.replace('[$,]', '', regex=True).astype(float)
                    
                    if chart_type == "Bar Chart":
                        fig = px.bar(plot_df, x=x_col, y=metric_for_chart,
                                    title=f"{metric_for_chart} by {x_col}",
                                    color=x_col)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Pie Chart":
                        fig = px.pie(plot_df, values=metric_for_chart, names=x_col,
                                    title=f"{metric_for_chart} Distribution by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Line Chart":
                        fig = px.line(plot_df, x=x_col, y=metric_for_chart,
                                     title=f"{metric_for_chart} Trend by {x_col}",
                                     markers=True)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Export functionality
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📥 Export Options")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # CSV Export
                csv_data = report_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name="custom_report.csv",
                    mime="text/csv"
                )
            
            with export_col2:
                st.info("To export charts as PNG, right-click on the chart and select 'Save as PNG'")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Natural Language Query (Simulated)
    st.subheader("🔎 Natural Language Query")
    
    query = st.text_input(
        "Ask a question about the portfolio (e.g., 'Show me all energy projects in Indonesia with delayed disbursements')",
        placeholder="Enter your query here..."
    )
    
    if query:
        st.info("""
        **Query Processing**: This is a simulated natural language query interface.
        
        In production, this would connect to an NLP engine that parses the query and 
        automatically generates the appropriate filters and visualizations.
        
        **Detected Intent**: Filter projects/loans based on criteria
        
        **Suggested Filters**:
        - Sector: Energy
        - Country: Indonesia  
        - Status: Check disbursement delays
        
        Try the filters above to see relevant results!
        """)


def render_portfolio_risk_tab(loans_df, projects_df):
    """Render the Portfolio Risk Dashboard tab."""
    
    st.header("⚠️ Portfolio Risk Dashboard")
    st.markdown("---")
    
    st.info("""
    **Risk Analytics**: This dashboard provides portfolio-level risk metrics including 
    Value at Risk (VaR), credit risk heatmap, maturity profile, and stress testing scenarios.
    """)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_exposure = loans_df["Outstanding Balance (USD)"].sum()
    high_risk_exposure = loans_df[
        loans_df["Risk Rating"].isin(["BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "D"])
    ]["Outstanding Balance (USD)"].sum()
    
    # Simple VaR calculation (parametric approach)
    returns_std = 0.15  # Assumed portfolio return volatility
    var_95 = total_exposure * 1.645 * returns_std
    var_99 = total_exposure * 2.326 * returns_std
    
    with col1:
        st.metric(
            label="Total Exposure",
            value=f"${total_exposure/1e9:.2f}B",
            delta=None
        )
    with col2:
        st.metric(
            label="High-Risk Exposure",
            value=f"${high_risk_exposure/1e9:.2f}B",
            delta=None
        )
    with col3:
        st.metric(
            label="VaR (95% Confidence)",
            value=f"${var_95/1e6:.1f}M",
            help="Maximum expected loss at 95% confidence level over 1-year horizon"
        )
    with col4:
        st.metric(
            label="VaR (99% Confidence)",
            value=f"${var_99/1e6:.1f}M",
            help="Maximum expected loss at 99% confidence level over 1-year horizon"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Credit Risk Heatmap
    st.subheader("🔥 Credit Risk Heatmap")
    
    merged_df = loans_df.merge(
        projects_df[['Project ID', 'Country', 'Sector']], 
        on='Project ID'
    )
    
    # Create risk score
    risk_score_map = {
        'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
        'A+': 5, 'A': 6, 'A-': 7, 'BBB+': 8, 'BBB': 9, 'BBB-': 10,
        'BB+': 11, 'BB': 12, 'BB-': 13, 'B+': 14, 'B': 15, 'B-': 16,
        'CCC+': 17, 'CCC': 18, 'CCC-': 19, 'D': 20
    }
    merged_df['Risk Score'] = merged_df['Risk Rating'].map(risk_score_map)
    
    # Aggregate by country and sector
    heatmap_data = merged_df.groupby(['Country', 'Sector']).agg({
        'Outstanding Balance (USD)': 'sum',
        'Risk Score': 'mean'
    }).reset_index()
    
    heatmap_pivot = heatmap_data.pivot(index='Country', columns='Sector', values='Risk Score')
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Sector", y="Country", color="Avg Risk Score"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        title="Credit Risk Heatmap by Country and Sector (Higher = Riskier)",
        color_continuous_scale=px.colors.sequential.YlOrRd
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Maturity Profile
    st.subheader("📅 Loan Maturity Profile")
    
    maturity_df = loans_df[loans_df["Maturity Date"].notna()].copy()
    maturity_df["Maturity Year"] = maturity_df["Maturity Date"].dt.year
    maturity_df["Years to Maturity"] = maturity_df["Maturity Year"] - datetime.now().year
    
    # Group by years to maturity
    maturity_buckets = maturity_df.copy()
    maturity_buckets['Maturity Bucket'] = maturity_buckets['Years to Maturity'].apply(
        lambda x: '0-1 Years' if x <= 1 else
                  '1-3 Years' if x <= 3 else
                  '3-5 Years' if x <= 5 else
                  '5-10 Years' if x <= 10 else '10+ Years'
    )
    
    bucket_order = ['0-1 Years', '1-3 Years', '3-5 Years', '5-10 Years', '10+ Years']
    maturity_summary = maturity_buckets.groupby('Maturity Bucket')['Outstanding Balance (USD)'].sum().reindex(bucket_order).reset_index()
    
    fig_maturity = px.bar(
        maturity_summary,
        x='Maturity Bucket',
        y='Outstanding Balance (USD)',
        title="Portfolio Maturity Profile",
        labels={'Maturity Bucket': 'Time to Maturity', 'Outstanding Balance (USD)': 'Outstanding Amount (USD)'},
        color='Outstanding Balance (USD)',
        color_continuous_scale=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_maturity, use_container_width=True)
    
    # Upcoming maturities
    upcoming = maturity_df[maturity_df["Years to Maturity"] <= 1].sort_values("Years to Maturity")
    if len(upcoming) > 0:
        st.warning(f"**{len(upcoming)} loans** maturing within the next 12 months totaling **${upcoming['Outstanding Balance (USD)'].sum()/1e6:.1f}M**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stress Testing
    st.subheader("🧪 Stress Testing Scenarios")
    
    st.markdown("Simulate impact of various stress scenarios on the portfolio:")
    
    stress_col1, stress_col2, stress_col3 = st.columns(3)
    
    base_exposure = total_exposure
    base_npl = loans_df[
        (loans_df["Covenant Status"] == "Breach") | 
        (loans_df["Risk Rating"].isin(["CCC+", "CCC", "CCC-", "D"]))
    ]["Outstanding Balance (USD)"].sum()
    
    with stress_col1:
        st.markdown("**Scenario 1: 10% Currency Depreciation**")
        if st.button("Run Scenario 1"):
            # Assume 50% of portfolio is in local currency exposure
            local_exposure = base_exposure * 0.5
            impact = local_exposure * 0.10
            new_npl = base_npl * 1.15  # NPLs increase due to currency stress
            
            st.metric("Portfolio Impact", f"-${impact/1e6:.1f}M")
            st.metric("New NPL Level", f"${new_npl/1e6:.1f}M")
            st.metric("NPL Ratio Change", f"+{(new_npl/base_exposure*100 - base_npl/base_exposure*100):.2f}%")
    
    with stress_col2:
        st.markdown("**Scenario 2: 2% Interest Rate Increase**")
        if st.button("Run Scenario 2"):
            # Higher rates affect non-sovereign borrowers more
            ns_exposure = loans_df[loans_df["Loan Type"] == "Non-Sovereign"]["Outstanding Balance (USD)"].sum()
            duration = 5  # Assumed average duration
            impact = ns_exposure * 0.02 * duration
            
            st.metric("Portfolio Impact", f"-${impact/1e6:.1f}M")
            st.metric("Affected Exposure", f"${ns_exposure/1e6:.1f}M")
    
    with stress_col3:
        st.markdown("**Scenario 3: Combined Stress**")
        if st.button("Run Scenario 3"):
            combined_impact = base_exposure * 0.08  # 8% combined impact
            new_npl = base_npl * 1.30
            
            st.metric("Portfolio Impact", f"-${combined_impact/1e6:.1f}M")
            st.metric("New NPL Level", f"${new_npl/1e6:.1f}M")
            st.metric("Capital Adequacy Impact", "-1.2%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk Ratings Distribution
    st.subheader("📊 Risk Rating Distribution")
    
    rating_dist = loans_df.groupby("Risk Rating")["Outstanding Balance (USD)"].sum().reset_index()
    rating_dist["Risk Category"] = rating_dist["Risk Rating"].apply(
        lambda x: 'Investment Grade' if x in ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-'] else 'Non-Investment Grade'
    )
    
    fig_rating = px.bar(
        rating_dist,
        x='Risk Rating',
        y='Outstanding Balance (USD)',
        title="Outstanding Balance by Risk Rating",
        color='Risk Category',
        color_discrete_map={'Investment Grade': '#27ae60', 'Non-Investment Grade': '#e74c3c'}
    )
    st.plotly_chart(fig_rating, use_container_width=True)


def main():
    """Main application function."""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/bank.png", width=80)
        st.title("MDB Monitor")
        st.markdown("---")
        
        navigation = st.radio(
            "Navigation",
            ["Project Overview", "Loan Portfolio", "Disbursement Monitoring", 
             "Data Quality & Reconciliation", "Business Intelligence", "Portfolio Risk"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        ### About This Dashboard
        
        This interactive dashboard simulates a **Multilateral Development Bank's** 
        project and loan data ecosystem.
        
        **Purpose**: Demonstrate working knowledge of:
        - Project lifecycle management
        - Loan portfolio monitoring
        - Disbursement tracking
        - Data quality analysis
        - Risk analytics
        
        Built for AIIB Digital Program Associate – Data Analysis role application.
        """)
        
        st.markdown("---")
        st.caption("© 2024 MDB Portfolio Monitor")
    
    # Load data
    data = load_data()
    projects_df = data["projects"]
    loans_df = data["loans"]
    disbursements_df = data["disbursements"]
    issues_df = data["issues"]
    completeness_df = data["completeness"]
    
    # Render tabs based on navigation
    if navigation == "Project Overview":
        render_project_overview_tab(projects_df, loans_df)
    elif navigation == "Loan Portfolio":
        render_loan_portfolio_tab(loans_df, projects_df)
    elif navigation == "Disbursement Monitoring":
        render_disbursement_monitoring_tab(disbursements_df, loans_df, projects_df)
    elif navigation == "Data Quality & Reconciliation":
        render_data_quality_tab(issues_df, completeness_df, projects_df, loans_df)
    elif navigation == "Business Intelligence":
        render_bi_self_service_tab(projects_df, loans_df)
    elif navigation == "Portfolio Risk":
        render_portfolio_risk_tab(loans_df, projects_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>MDB Project & Loan Portfolio Monitor</strong></p>
        <p>A demonstration dashboard showcasing data analysis capabilities for multilateral development banking operations.</p>
        <p>Built with Streamlit | Data generated for demonstration purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
