# MDB Project & Loan Portfolio Monitor

An interactive Streamlit dashboard simulating a multilateral development bank's project and loan data ecosystem. This portfolio piece demonstrates working knowledge of project and loan products for the AIIB Digital Program Associate – Data Analysis role.

## 🏦 Overview

This dashboard provides comprehensive monitoring capabilities for:
- **Project Portfolio**: Track infrastructure projects across sectors and countries
- **Loan Book**: Monitor commitments, disbursements, and outstanding balances
- **Disbursement Tracking**: Schedule and monitor loan disbursements
- **Data Quality**: Identify and resolve data quality issues
- **Business Intelligence**: Build custom reports and visualizations
- **Risk Analytics**: Portfolio risk metrics and stress testing

## 🚀 Quick Start

### Running in GitHub Codespaces

1. Open this repository in a GitHub Codespace
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. The dashboard will open automatically in your browser

### Running Locally

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📊 Features

### 1. Project Overview Tab
- View all infrastructure projects with key details
- Filter by sector, country, and status
- KPI cards showing portfolio value, active projects, and disbursement ratio
- Visual breakdowns by sector, country, and project status

### 2. Loan Portfolio Tab
- Complete loan book with commitment, disbursed, and outstanding amounts
- Classification breakdowns by loan type, risk rating, and covenant status
- Portfolio concentration analysis by country and sector
- Non-performing loan (NPL) tracker

### 3. Disbursement Monitoring Tab
- Disbursement schedule with status tracking
- Delayed disbursement alerts with delay metrics
- Progress bars showing disbursed vs. committed amounts
- Monthly disbursement trend analysis

### 4. Data Quality & Reconciliation Tab
- Automated detection of common data quality issues:
  - Mismatched project cost vs. loan amounts
  - Missing covenant data
  - Duplicate project entries
  - Classification inconsistencies
- Severity-based issue flagging (High/Medium/Low)
- Data completeness scores per project
- Reconciliation log with resolution actions

### 5. Business Intelligence & Self-Service Tab
- Custom report builder with selectable dimensions and metrics
- Dynamic chart generation (bar, pie, line charts)
- CSV export functionality
- Simulated natural language query interface

### 6. Portfolio Risk Dashboard Tab
- Value at Risk (VaR) calculations at 95% and 99% confidence
- Credit risk heatmap by country and sector
- Loan maturity profile with upcoming maturities alert
- Stress testing scenarios:
  - 10% currency depreciation
  - 2% interest rate increase
  - Combined stress scenario

## 🎨 Design

The dashboard uses a professional color scheme aligned with financial institution branding:
- **Primary Color**: Dark Blue (#1a3a5c)
- **Accent Color**: Gold (#d4a843)
- **Background**: Light Gray (#f5f5f5)

## 📁 File Structure

```
├── app.py                 # Main Streamlit application
├── data_generator.py      # Synthetic data generation module
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Data Manipulation**: pandas, numpy
- **Visualization**: Plotly
- **Data Generation**: Custom synthetic data generator with seed 42 for reproducibility

## 📈 Data Specifications

The dashboard generates realistic MDB data following these rules:
- **25 projects** across 8 Asian countries
- **Sectors**: Energy (8), Transport (7), Water (5), Digital Infrastructure (5)
- **Status Distribution**: Concept (5), Appraisal (5), Implementation (10), Completion (5)
- **Loan Types**: 60% Sovereign, 40% Non-Sovereign
- **Risk Ratings**: 50% AA or above, 30% A-BBB, 15% BB-B, 5% CCC or below
- **Deliberate Data Quality Issues**: 6 issues for demonstration purposes

## 💼 Job Application Context

This dashboard was created to demonstrate competencies for the **AIIB Digital Program Associate – Data Analysis** role, specifically addressing:

1. **Project and Loan Product Knowledge**: Understanding of MDB financing instruments
2. **Data Analysis Skills**: Filtering, aggregation, and visualization
3. **Data Quality Management**: Root cause analysis of data issues
4. **Risk Analytics**: Portfolio risk assessment and stress testing
5. **Business Intelligence**: Self-service reporting capabilities

## 📝 Notes

- All data is synthetically generated for demonstration purposes
- No external data files are required
- The application runs entirely in a browser with no backend server needed
- Data is reproducible using seed 42

## 📄 License

This project is provided as-is for portfolio demonstration purposes.

---

*Built with ❤️ for AIIB Digital Program Associate application*
