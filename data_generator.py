"""
Data Generator for MDB Project & Loan Portfolio Monitor
Generates synthetic but realistic data for multilateral development bank projects and loans.
Uses seed 42 for reproducibility.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
COUNTRIES = ["Indonesia", "India", "Bangladesh", "Pakistan", "Vietnam", "Philippines", "Thailand", "Mongolia"]
SECTORS = ["Energy", "Transport", "Water", "Digital Infrastructure"]
SECTOR_DISTRIBUTION = [8, 7, 5, 5]  # Number of projects per sector
STATUSES = ["Concept", "Appraisal", "Implementation", "Completion"]
STATUS_DISTRIBUTION = [5, 5, 10, 5]
LOAN_TYPES = ["Sovereign", "Non-Sovereign"]
COFINANCIERS = ["World Bank", "Asian Development Bank", "Japan International Cooperation Agency", 
                "Korea Development Bank", "European Investment Bank", "None"]
RISK_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", 
                "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "D"]
COVENANT_STATUSES = ["Compliant", "Watch", "Breach"]

def generate_projects(n_projects=25):
    """Generate project data with realistic distributions."""
    
    # Build sector list based on distribution
    sectors = []
    for sector, count in zip(SECTORS, SECTOR_DISTRIBUTION):
        sectors.extend([sector] * count)
    
    # Build status list based on distribution  
    statuses = []
    for status, count in zip(STATUSES, STATUS_DISTRIBUTION):
        statuses.extend([status] * count)
    
    # Shuffle to randomize order
    random.shuffle(sectors)
    random.shuffle(statuses)
    
    projects = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(n_projects):
        project_id = f"AIIB-{2020 + (i % 5):02d}-{i+1:03d}"
        country = random.choice(COUNTRIES)
        sector = sectors[i] if i < len(sectors) else random.choice(SECTORS)
        status = statuses[i] if i < len(statuses) else random.choice(STATUSES)
        
        # Generate realistic project costs based on sector
        if sector == "Energy":
            total_cost = np.random.uniform(200_000_000, 1_500_000_000)
        elif sector == "Transport":
            total_cost = np.random.uniform(300_000_000, 2_000_000_000)
        elif sector == "Water":
            total_cost = np.random.uniform(100_000_000, 800_000_000)
        else:  # Digital Infrastructure
            total_cost = np.random.uniform(50_000_000, 500_000_000)
        
        # AIIB typically finances 20-50% of project cost
        aiib_share = np.random.uniform(0.2, 0.5)
        aiib_loan = total_cost * aiib_share
        
        # Co-financing
        cofinancier = random.choice(COFINANCIERS)
        if cofinancier != "None":
            cofinancing = total_cost * np.random.uniform(0.15, 0.35)
        else:
            cofinancing = 0
            cofinancier = None
        
        # Dates based on status
        if status == "Concept":
            approval_date = None
            completion_date = None
        elif status == "Appraisal":
            approval_date = base_date + timedelta(days=random.randint(1000, 1400))
            completion_date = None
        elif status == "Implementation":
            approval_date = base_date + timedelta(days=random.randint(500, 1000))
            completion_date = None
        else:  # Completion
            approval_date = base_date + timedelta(days=random.randint(300, 800))
            completion_date = approval_date + timedelta(days=random.randint(700, 1500))
        
        project_name = generate_project_name(country, sector, i)
        
        projects.append({
            "Project ID": project_id,
            "Project Name": project_name,
            "Country": country,
            "Sector": sector,
            "Status": status,
            "Total Project Cost (USD)": round(total_cost, 0),
            "AIIB Loan Amount (USD)": round(aiib_loan, 0),
            "Co-financing Amount (USD)": round(cofinancing, 0),
            "Co-financier": cofinancier,
            "Approval Date": approval_date,
            "Completion Date": completion_date
        })
    
    return pd.DataFrame(projects)


def generate_project_name(country, sector, idx):
    """Generate realistic project names."""
    prefixes = {
        "Energy": ["Renewable Energy", "Solar Power", "Wind Farm", "Hydropower", "Grid Expansion", "Clean Energy"],
        "Transport": ["Highway", "Railway", "Port Modernization", "Airport Expansion", "Urban Transit", "Bridge Construction"],
        "Water": ["Water Supply", "Sanitation", "Flood Control", "Irrigation", "Wastewater Treatment"],
        "Digital Infrastructure": ["Broadband Network", "Digital Connectivity", "5G Infrastructure", "Data Center", "Smart City"]
    }
    
    suffixes = ["Project", "Program", "Initiative", "Development Project", "Infrastructure Project"]
    
    prefix = random.choice(prefixes[sector])
    suffix = random.choice(suffixes)
    
    return f"{country} {prefix} {suffix} {idx+1}"


def generate_loans(projects_df):
    """Generate loan data linked to projects."""
    
    loans = []
    base_date = datetime(2020, 1, 1)
    
    for _, project in projects_df.iterrows():
        if project["AIIB Loan Amount (USD)"] > 0:
            loan_id = f"LN-{project['Project ID'].split('-')[1]}-{project['Project ID'].split('-')[2]}"
            
            # Loan type: 60% Sovereign, 40% Non-Sovereign
            loan_type = "Sovereign" if random.random() < 0.6 else "Non-Sovereign"
            
            commitment = project["AIIB Loan Amount (USD)"]
            
            # Disbursement ratio between 20% and 95%, depending on status
            if project["Status"] == "Concept":
                disbursed_ratio = 0
            elif project["Status"] == "Appraisal":
                disbursed_ratio = np.random.uniform(0, 0.1)
            elif project["Status"] == "Implementation":
                disbursed_ratio = np.random.uniform(0.3, 0.8)
            else:  # Completion
                disbursed_ratio = np.random.uniform(0.85, 0.98)
            
            disbursed = commitment * disbursed_ratio
            outstanding = commitment - disbursed
            
            # Interest rate based on loan type and risk
            base_rate = 2.5 if loan_type == "Sovereign" else 4.0
            interest_rate = base_rate + np.random.uniform(0.5, 2.5)
            
            # Maturity date (5-20 years from approval)
            if project["Approval Date"]:
                maturity_years = random.randint(5, 20)
                maturity_date = project["Approval Date"] + timedelta(days=maturity_years * 365)
            else:
                maturity_date = None
            
            # Risk rating distribution: 50% AA or above, 30% A-BBB, 15% BB-B, 5% CCC or below
            rand = random.random()
            if rand < 0.5:
                risk_rating = random.choice(["AAA", "AA+", "AA", "AA-"])
            elif rand < 0.8:
                risk_rating = random.choice(["A+", "A", "A-", "BBB+", "BBB", "BBB-"])
            elif rand < 0.95:
                risk_rating = random.choice(["BB+", "BB", "BB-", "B+", "B", "B-"])
            else:
                risk_rating = random.choice(["CCC+", "CCC", "CCC-", "D"])
            
            # Covenant status - correlate with risk rating
            if risk_rating in ["CCC+", "CCC", "CCC-", "D"]:
                covenant_status = random.choices(["Compliant", "Watch", "Breach"], weights=[0.2, 0.3, 0.5])[0]
            elif risk_rating.startswith("B"):
                covenant_status = random.choices(["Compliant", "Watch", "Breach"], weights=[0.4, 0.4, 0.2])[0]
            else:
                covenant_status = random.choices(["Compliant", "Watch", "Breach"], weights=[0.8, 0.15, 0.05])[0]
            
            loans.append({
                "Loan ID": loan_id,
                "Project ID": project["Project ID"],
                "Loan Type": loan_type,
                "Commitment Amount (USD)": round(commitment, 0),
                "Disbursed Amount (USD)": round(disbursed, 0),
                "Outstanding Balance (USD)": round(outstanding, 0),
                "Interest Rate (%)": round(interest_rate, 2),
                "Maturity Date": maturity_date,
                "Covenant Status": covenant_status,
                "Risk Rating": risk_rating,
                "Currency": "USD"  # Simplified - all USD for now
            })
    
    return pd.DataFrame(loans)


def generate_disbursements(loans_df, projects_df):
    """Generate disbursement schedule for each loan."""
    
    disbursements = []
    
    for _, loan in loans_df.iterrows():
        project = projects_df[projects_df["Project ID"] == loan["Project ID"]].iloc[0]
        
        if loan["Commitment Amount (USD)"] <= 0:
            continue
            
        # Generate 4-8 disbursement tranches
        n_tranches = random.randint(4, 8)
        tranche_amounts = np.array_split(np.linspace(loan["Commitment Amount (USD)"] / n_tranches, 
                                                      loan["Commitment Amount (USD)"] / n_tranches * 1.5, 
                                                      n_tranches), n_tranches)
        
        # Normalize to match commitment
        total = sum([t[0] for t in tranche_amounts])
        tranche_amounts = [t[0] * loan["Commitment Amount (USD)"] / total for t in tranche_amounts]
        
        cumulative_disbursed = 0
        start_date = project["Approval Date"] if project["Approval Date"] else datetime(2022, 1, 1)
        
        for i, amount in enumerate(tranche_amounts):
            scheduled_date = start_date + timedelta(days=i * 90 + random.randint(-15, 15))
            
            # Determine if this tranche has been disbursed
            if cumulative_disbursed + amount <= loan["Disbursed Amount (USD)"]:
                actual_date = scheduled_date + timedelta(days=random.randint(-10, 30))
                status = "Disbursed"
            elif cumulative_disbursed < loan["Disbursed Amount (USD)"]:
                # Partial disbursement
                actual_date = scheduled_date + timedelta(days=random.randint(0, 45))
                status = "Disbursed"
                amount = loan["Disbursed Amount (USD)"] - cumulative_disbursed
            else:
                actual_date = None
                if scheduled_date < datetime.now():
                    status = random.choices(["Delayed", "Pending"], weights=[0.3, 0.7])[0]
                else:
                    status = "Pending"
            
            disbursements.append({
                "Disbursement ID": f"DISB-{loan['Loan ID']}-{i+1:02d}",
                "Loan ID": loan["Loan ID"],
                "Project ID": loan["Project ID"],
                "Tranche Number": i + 1,
                "Scheduled Date": scheduled_date,
                "Actual Date": actual_date,
                "Amount (USD)": round(amount, 0),
                "Status": status
            })
            
            cumulative_disbursed += amount
    
    return pd.DataFrame(disbursements)


def introduce_data_quality_issues(projects_df, loans_df, disbursements_df):
    """Introduce deliberate data quality issues for the Data Quality tab."""
    
    issues = []
    
    # Issue 1: Mismatched project cost vs loan amounts (High severity)
    mismatched_idx = 2
    original_loan = loans_df.loc[loans_df["Project ID"] == projects_df.iloc[mismatched_idx]["Project ID"], 
                                  "Commitment Amount (USD)"].values[0]
    loans_df.loc[loans_df["Project ID"] == projects_df.iloc[mismatched_idx]["Project ID"], 
                   "Commitment Amount (USD)"] = original_loan * 1.15  # 15% mismatch
    
    issues.append({
        "Issue ID": "DQ-001",
        "Type": "Data Mismatch",
        "Description": f"Loan commitment exceeds project AIIB loan amount by 15% for {projects_df.iloc[mismatched_idx]['Project ID']}",
        "Affected Entity": projects_df.iloc[mismatched_idx]["Project ID"],
        "Severity": "High",
        "Status": "Unresolved"
    })
    
    # Issue 2: Missing covenant data (Medium severity)
    missing_covenant_idx = 5
    loans_df.loc[loans_df["Project ID"] == projects_df.iloc[missing_covenant_idx]["Project ID"], 
                   "Covenant Status"] = None
    
    issues.append({
        "Issue ID": "DQ-002",
        "Type": "Missing Data",
        "Description": f"Missing covenant status for loan linked to {projects_df.iloc[missing_covenant_idx]['Project ID']}",
        "Affected Entity": loans_df[loans_df["Project ID"] == projects_df.iloc[missing_covenant_idx]["Project ID"]]["Loan ID"].values[0],
        "Severity": "Medium",
        "Status": "Unresolved"
    })
    
    # Issue 3: Duplicate project entry simulation (High severity)
    duplicate_project = projects_df.iloc[0].copy()
    duplicate_project["Project ID"] = duplicate_project["Project ID"] + "-DUP"
    duplicate_project["Project Name"] = duplicate_project["Project Name"] + " (Duplicate)"
    
    issues.append({
        "Issue ID": "DQ-003",
        "Type": "Duplicate Record",
        "Description": f"Potential duplicate of {projects_df.iloc[0]['Project ID']} detected",
        "Affected Entity": duplicate_project["Project ID"],
        "Severity": "High",
        "Status": "Under Review"
    })
    
    # Issue 4: Inconsistent country classification (Medium severity)
    inconsistent_idx = 10
    original_country = projects_df.iloc[inconsistent_idx]["Country"]
    # We'll flag this but not change it - just note the issue
    
    issues.append({
        "Issue ID": "DQ-004",
        "Type": "Classification Inconsistency",
        "Description": f"Country classification for {projects_df.iloc[inconsistent_idx]['Project ID']} differs from borrower registration country",
        "Affected Entity": projects_df.iloc[inconsistent_idx]["Project ID"],
        "Severity": "Medium",
        "Status": "Resolved"
    })
    
    # Issue 5: Missing maturity date (Low severity)
    missing_maturity_idx = 3
    loans_df.loc[loans_df["Project ID"] == projects_df.iloc[missing_maturity_idx]["Project ID"], 
                   "Maturity Date"] = None
    
    issues.append({
        "Issue ID": "DQ-005",
        "Type": "Missing Data",
        "Description": f"Missing maturity date for loan {loans_df[loans_df['Project ID'] == projects_df.iloc[missing_maturity_idx]['Project ID']]['Loan ID'].values[0]}",
        "Affected Entity": loans_df[loans_df["Project ID"] == projects_df.iloc[missing_maturity_idx]["Project ID"]]["Loan ID"].values[0],
        "Severity": "Low",
        "Status": "Unresolved"
    })
    
    # Issue 6: Negative disbursement amount anomaly (High severity)
    if len(disbursements_df) > 0:
        anomalies_idx = 7
        if anomalies_idx < len(disbursements_df):
            disbursements_df.loc[anomalies_idx, "Amount (USD)"] = abs(disbursements_df.loc[anomalies_idx, "Amount (USD)"])
            issues.append({
                "Issue ID": "DQ-006",
                "Type": "Data Anomaly",
                "Description": f"Disbursement amount sign inconsistency detected for {disbursements_df.loc[anomalies_idx, 'Disbursement ID']}",
                "Affected Entity": disbursements_df.loc[anomalies_idx, "Disbursement ID"],
                "Severity": "High",
                "Status": "Unresolved"
            })
    
    issues_df = pd.DataFrame(issues)
    
    return projects_df, loans_df, disbursements_df, issues_df, duplicate_project


def calculate_data_completeness(projects_df, loans_df):
    """Calculate data completeness score per project."""
    
    completeness_scores = []
    
    for _, project in projects_df.iterrows():
        required_fields = ["Project ID", "Project Name", "Country", "Sector", "Status", 
                          "Total Project Cost (USD)", "AIIB Loan Amount (USD)"]
        optional_fields = ["Co-financing Amount (USD)", "Co-financier", "Approval Date", "Completion Date"]
        
        filled_required = sum(1 for field in required_fields if pd.notna(project.get(field)))
        filled_optional = sum(1 for field in optional_fields if pd.notna(project.get(field)))
        
        # Weight: required fields worth 70%, optional worth 30%
        score = (filled_required / len(required_fields)) * 70 + (filled_optional / len(optional_fields)) * 30
        
        # Check linked loan data
        loan = loans_df[loans_df["Project ID"] == project["Project ID"]]
        if len(loan) > 0:
            loan_fields = ["Loan ID", "Loan Type", "Commitment Amount (USD)", "Disbursed Amount (USD)",
                          "Outstanding Balance (USD)", "Interest Rate (%)", "Maturity Date", 
                          "Covenant Status", "Risk Rating"]
            filled_loan = sum(1 for field in loan_fields if pd.notna(loan.iloc[0].get(field)))
            loan_score = (filled_loan / len(loan_fields)) * 20  # Additional 20% from loan data
            score = min(100, score + loan_score)
        
        completeness_scores.append({
            "Project ID": project["Project ID"],
            "Completeness Score (%)": round(score, 1)
        })
    
    return pd.DataFrame(completeness_scores)


def generate_all_data():
    """Main function to generate all data."""
    
    # Generate base data
    projects_df = generate_projects(25)
    loans_df = generate_loans(projects_df)
    disbursements_df = generate_disbursements(loans_df, projects_df)
    
    # Introduce data quality issues
    projects_df, loans_df, disbursements_df, issues_df, duplicate_project = introduce_data_quality_issues(
        projects_df, loans_df, disbursements_df
    )
    
    # Add duplicate project for display purposes
    projects_with_dup = pd.concat([projects_df, pd.DataFrame([duplicate_project])], ignore_index=True)
    
    # Calculate completeness scores
    completeness_df = calculate_data_completeness(projects_df, loans_df)
    
    return {
        "projects": projects_df,
        "loans": loans_df,
        "disbursements": disbursements_df,
        "issues": issues_df,
        "completeness": completeness_df
    }


if __name__ == "__main__":
    data = generate_all_data()
    print(f"Generated {len(data['projects'])} projects")
    print(f"Generated {len(data['loans'])} loans")
    print(f"Generated {len(data['disbursements'])} disbursements")
    print(f"Generated {len(data['issues'])} data quality issues")
