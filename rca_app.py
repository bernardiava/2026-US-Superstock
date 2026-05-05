"""
Interactive Root Cause Analysis App
Uses top consulting frameworks (McKinsey, Bain, BCG) for structured problem-solving.
Auto-detects appropriate framework and guides users step-by-step.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json

# Page configuration
st.set_page_config(
    page_title="RCA Pro - Interactive Root Cause Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .framework-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .step-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Framework definitions used by top consulting firms
FRAMEWORKS = {
    "MECE": {
        "name": "MECE Principle",
        "firm": "McKinsey & Company",
        "description": "Mutually Exclusive, Collectively Exhaustive - ensures comprehensive and non-overlapping analysis",
        "best_for": ["Problem structuring", "Issue trees", "Categorization"],
        "steps": [
            "Define the core problem clearly",
            "Break down into mutually exclusive components",
            "Ensure all components are collectively exhaustive",
            "Prioritize branches based on impact",
            "Analyze each branch systematically"
        ]
    },
    "Five Whys": {
        "name": "Five Whys",
        "firm": "Toyota / Bain & Company",
        "description": "Iterative interrogative technique to explore cause-and-effect relationships",
        "best_for": ["Simple problems", "Process issues", "Quick diagnosis"],
        "steps": [
            "State the problem clearly",
            "Ask 'Why did this happen?'",
            "For each answer, ask 'Why?' again",
            "Continue until reaching root cause (typically 5 levels)",
            "Develop countermeasures for root causes"
        ]
    },
    "Issue Tree": {
        "name": "Issue Tree",
        "firm": "McKinsey & Company",
        "description": "Hierarchical breakdown of problems into sub-issues using MECE principle",
        "best_for": ["Complex problems", "Strategic analysis", "Team collaboration"],
        "steps": [
            "Define the overarching question",
            "Create first-level branches (key hypotheses)",
            "Break down each branch into sub-issues",
            "Continue until actionable level reached",
            "Prioritize and assign analysis workstreams"
        ]
    },
    "Hypothesis Driven": {
        "name": "Hypothesis-Driven Problem Solving",
        "firm": "McKinsey & Company",
        "description": "Start with potential solutions and test them systematically",
        "best_for": ["Time-constrained projects", "Experienced teams", "Data-rich environments"],
        "steps": [
            "Formulate initial hypothesis",
            "Design tests to validate/falsify",
            "Gather data efficiently",
            "Refine or pivot hypothesis",
            "Develop recommendations based on findings"
        ]
    },
    "Root Cause Matrix": {
        "name": "Root Cause Matrix",
        "firm": "Bain & Company",
        "description": "Systematic categorization of causes across multiple dimensions",
        "best_for": ["Multi-factorial problems", "Cross-functional issues", "Complex systems"],
        "steps": [
            "Identify symptom/problem statement",
            "Map potential cause categories (People, Process, Technology, etc.)",
            "Brainstorm specific causes within each category",
            "Score causes by likelihood and impact",
            "Focus investigation on high-priority causes"
        ]
    },
    "BCG Growth Share": {
        "name": "BCG Matrix Adaptation",
        "firm": "Boston Consulting Group",
        "description": "Prioritization framework adapted for root cause analysis",
        "best_for": ["Portfolio of issues", "Resource allocation", "Strategic prioritization"],
        "steps": [
            "List all identified potential causes",
            "Assess impact of each cause on the problem",
            "Assess ease of addressing each cause",
            "Plot on Impact vs. Effort matrix",
            "Prioritize quick wins and major impacts"
        ]
    },
    "Fishbone": {
        "name": "Fishbone (Ishikawa) Diagram",
        "firm": "Bain / Quality Management",
        "description": "Visual tool to organize potential causes into categories",
        "best_for": ["Manufacturing issues", "Process problems", "Team brainstorming"],
        "steps": [
            "Define the problem (effect) at the head",
            "Identify major categories (6Ms: Man, Machine, Method, Material, Measurement, Mother Nature)",
            "Brainstorm causes within each category",
            "Drill down into sub-causes",
            "Identify most likely root causes for investigation"
        ]
    },
    "Pareto": {
        "name": "Pareto Analysis (80/20 Rule)",
        "firm": "BCG / Bain",
        "description": "Focus on the vital few causes that create most of the problem",
        "best_for": ["Data-rich problems", "Prioritization", "Quick wins"],
        "steps": [
            "List all potential causes",
            "Quantify impact of each cause",
            "Rank causes by impact",
            "Calculate cumulative percentage",
            "Focus on top 20% causing 80% of problems"
        ]
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'selected_framework' not in st.session_state:
        st.session_state.selected_framework = None
    if 'problem_statement' not in st.session_state:
        st.session_state.problem_statement = ""
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}
    if 'five_whys_answers' not in st.session_state:
        st.session_state.five_whys_answers = {}
    if 'issue_tree_data' not in st.session_state:
        st.session_state.issue_tree_data = {}
    if 'fishbone_categories' not in st.session_state:
        st.session_state.fishbone_categories = {}
    if 'completed' not in st.session_state:
        st.session_state.completed = False

def detect_framework(problem_text: str) -> str:
    """Auto-detect the best framework based on problem characteristics"""
    problem_lower = problem_text.lower()
    
    # Keywords mapping to frameworks
    indicators = {
        "Five Whys": ["why", "simple", "quick", "process failure", "error", "mistake"],
        "Issue Tree": ["complex", "multiple factors", "strategic", "break down", "structure"],
        "MECE": ["categorize", "organize", "structure", "comprehensive", "overlapping"],
        "Hypothesis Driven": ["hypothesis", "test", "validate", "assume", "believe"],
        "Root Cause Matrix": ["multiple causes", "cross-functional", "departments", "factors"],
        "BCG Growth Share": ["prioritize", "resources", "multiple issues", "portfolio"],
        "Fishbone": ["manufacturing", "production", "quality", "defect", "6m"],
        "Pareto": ["data", "numbers", "frequency", "80/20", "majority", "minority"]
    }
    
    scores = {fw: 0 for fw in FRAMEWORKS.keys()}
    
    for framework, keywords in indicators.items():
        for keyword in keywords:
            if keyword in problem_lower:
                scores[framework] += 1
    
    # Return framework with highest score, default to Issue Tree for complex problems
    if max(scores.values()) == 0:
        return "Issue Tree"
    
    return max(scores, key=scores.get)

def render_homepage():
    """Render the homepage with framework selection"""
    st.markdown('<p class="main-header">🔍 RCA Pro - Interactive Root Cause Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Professional-grade root cause analysis using frameworks from McKinsey, Bain, and BCG
    
    This app automatically detects the best framework for your problem and guides you through 
    a structured, step-by-step analysis process.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Start Your Analysis")
        problem_input = st.text_area(
            "Describe your problem in detail:",
            height=150,
            placeholder="Example: Our customer satisfaction scores have dropped 15% in Q3, particularly in the Northeast region. Customer complaints mention long wait times and unresolved issues on first contact...",
            key="problem_input"
        )
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if problem_input.strip():
                st.session_state.problem_statement = problem_input
                detected_fw = detect_framework(problem_input)
                st.session_state.selected_framework = detected_fw
                st.session_state.current_step = 0
                st.rerun()
            else:
                st.warning("Please describe your problem to continue.")
    
    with col2:
        st.subheader("🎯 Available Frameworks")
        st.info(f"**Auto-detected**: Will be determined based on your problem description")
        
        for fw_key, fw_data in FRAMEWORKS.items():
            with st.expander(f"**{fw_data['name']}** ({fw_data['firm']})"):
                st.write(fw_data["description"])
                st.write("**Best for:** " + ", ".join(fw_data["best_for"]))
    
    # Show framework comparison
    st.markdown("---")
    st.subheader("📊 Framework Comparison")
    
    framework_df = pd.DataFrame([
        {
            "Framework": data["name"],
            "Firm": data["firm"],
            "Best For": ", ".join(data["best_for"]),
            "Steps": len(data["steps"])
        }
        for data in FRAMEWORKS.values()
    ])
    
    st.dataframe(framework_df, use_container_width=True, hide_index=True)

def render_five_whys():
    """Render Five Whys analysis interface"""
    st.subheader("🔍 Five Whys Analysis")
    st.write(f"**Problem:** {st.session_state.problem_statement}")
    
    answers = st.session_state.five_whys_answers
    
    for i in range(1, 6):
        st.markdown(f"**Why #{i}:**")
        if i == 1:
            question = "Why did this problem occur?"
        else:
            prev_answer = answers.get(i-1, "")
            question = f"Why did '{prev_answer}' happen?" if prev_answer else f"Why did the previous cause occur?"
        
        answer = st.text_area(
            question,
            value=answers.get(i, ""),
            key=f"why_{i}",
            height=80,
            help="Press Enter or click outside to save"
        )
        
        if answer != answers.get(i, ""):
            st.session_state.five_whys_answers[i] = answer
            st.rerun()
        
        if i < 5 and answers.get(i):
            st.markdown("---")
    
    # Root cause summary
    if all(answers.get(i) for i in range(1, 6)):
        st.success("✅ Five Whys completed! Review your root cause chain below:")
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=[1] * 5,
            mode="markers+text+lines",
            marker=dict(size=20, color="#1f77b4"),
            text=[f"Why {i}: {answers[i][:30]}..." for i in range(1, 6)],
            textposition="top center",
            line=dict(width=2, dash="dash")
        ))
        
        fig.update_layout(
            height=200,
            showlegend=False,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            title="Root Cause Chain"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💡 Recommended Actions")
        root_cause = answers.get(5, "")
        st.text_area("Based on the root cause identified, what countermeasures will you implement?",
                    height=100, key="countermeasures")
        
        if st.button("✓ Complete Analysis"):
            st.session_state.completed = True
            st.rerun()

def render_issue_tree():
    """Render Issue Tree analysis interface"""
    st.subheader("🌳 Issue Tree Analysis")
    st.write(f"**Core Question:** {st.session_state.problem_statement}")
    
    tree_data = st.session_state.issue_tree_data
    
    # Level 1 branches
    st.markdown("### Level 1: Main Hypotheses")
    num_branches = st.number_input("Number of main branches:", min_value=2, max_value=5, value=3, key="num_branches")
    
    for i in range(int(num_branches)):
        branch_key = f"branch_{i}"
        branch_name = st.text_input(f"Branch {i+1}:", value=tree_data.get(branch_key, {}).get("name", ""), 
                                   key=f"input_{branch_key}", help="Make sure branches are MECE")
        
        if branch_name:
            if branch_key not in tree_data:
                tree_data[branch_key] = {"name": branch_name, "sub_issues": []}
            else:
                tree_data[branch_key]["name"] = branch_name
            
            # Sub-issues for each branch
            st.markdown(f"**Sub-issues for '{branch_name}':**")
            num_sub = st.number_input(f"Number of sub-issues:", min_value=0, max_value=5, 
                                     value=len(tree_data.get(branch_key, {}).get("sub_issues", [])),
                                     key=f"num_sub_{i}")
            
            sub_issues = []
            for j in range(int(num_sub)):
                sub_key = f"{branch_key}_sub_{j}"
                sub_issue = st.text_input(f"  Sub-issue {j+1}:", value=tree_data.get(branch_key, {}).get("sub_issues", [{}]*num_sub)[j].get("name", "") if len(tree_data.get(branch_key, {}).get("sub_issues", [])) > j else "",
                                         key=f"input_{sub_key}")
                if sub_issue:
                    sub_issues.append({"name": sub_issue})
            
            tree_data[branch_key]["sub_issues"] = sub_issues
        
        st.markdown("---")
    
    st.session_state.issue_tree_data = tree_data
    
    # Visualization
    if tree_data:
        st.subheader("📊 Issue Tree Visualization")
        
        # Create a simple tree visualization
        fig = go.Figure()
        
        nodes_x = []
        nodes_y = []
        node_labels = []
        
        # Root node
        nodes_x.append(0.5)
        nodes_y.append(1.0)
        node_labels.append("Core Problem")
        
        # Level 1 nodes
        branch_count = len([k for k in tree_data.keys() if k.startswith("branch_") and tree_data[k].get("name")])
        if branch_count > 0:
            spacing = 0.8 / branch_count
            for idx, branch_key in enumerate([k for k in tree_data.keys() if k.startswith("branch_")][:branch_count]):
                branch_name = tree_data[branch_key].get("name", "")
                if branch_name:
                    nodes_x.append(0.1 + idx * spacing)
                    nodes_y.append(0.6)
                    node_labels.append(branch_name[:20])
                    
                    # Level 2 nodes (sub-issues)
                    sub_issues = tree_data[branch_key].get("sub_issues", [])
                    if sub_issues:
                        sub_spacing = spacing / (len(sub_issues) + 1)
                        for s_idx, sub in enumerate(sub_issues):
                            sub_name = sub.get("name", "")
                            if sub_name:
                                nodes_x.append(0.1 + idx * spacing - spacing/2 + (s_idx+1) * sub_spacing)
                                nodes_y.append(0.2)
                                node_labels.append(sub_name[:15])
        
        fig.add_trace(go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers+text",
            marker=dict(size=30, color="#1f77b4"),
            text=node_labels,
            textposition="bottom center"
        ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(range=[0, 1], showgrid=False),
            yaxis=dict(range=[0, 1.1], showgrid=False),
            title="Issue Tree Structure"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("✓ Complete Issue Tree Analysis"):
        st.session_state.completed = True
        st.rerun()

def render_fishbone():
    """Render Fishbone (Ishikawa) Diagram interface"""
    st.subheader("🐟 Fishbone (Ishikawa) Diagram")
    st.write(f"**Problem (Effect):** {st.session_state.problem_statement}")
    
    # Standard 6M categories
    categories = ["Man (People)", "Machine", "Method", "Material", "Measurement", "Mother Nature (Environment)"]
    
    fishbone_data = st.session_state.fishbone_categories
    
    cols = st.columns(2)
    col_idx = 0
    
    for category in categories:
        with cols[col_idx % 2]:
            st.markdown(f"**{category}**")
            causes = st.text_area(
                f"Causes related to {category}:",
                value=fishbone_data.get(category, ""),
                height=100,
                key=f"fishbone_{category}",
                help="List each cause on a new line"
            )
            fishbone_data[category] = causes
        col_idx += 1
    
    st.session_state.fishbone_categories = fishbone_data
    
    # Visualization
    if any(fishbone_data.values()):
        st.subheader("📊 Fishbone Diagram")
        
        # Count causes per category
        cause_counts = {}
        for cat, causes_text in fishbone_data.items():
            if causes_text.strip():
                cause_counts[cat.split()[0]] = len([c for c in causes_text.split('\n') if c.strip()])
            else:
                cause_counts[cat.split()[0]] = 0
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(cause_counts.keys()),
                y=list(cause_counts.values()),
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title="Causes by Category",
            xaxis_title="Category",
            yaxis_title="Number of Causes",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("✓ Complete Fishbone Analysis"):
        st.session_state.completed = True
        st.rerun()

def render_pareto():
    """Render Pareto Analysis interface"""
    st.subheader("📊 Pareto Analysis (80/20 Rule)")
    st.write(f"**Problem:** {st.session_state.problem_statement}")
    
    # Data input
    st.markdown("### Enter Cause Data")
    
    pareto_data = st.session_state.analysis_data.get('pareto', [])
    
    num_causes = st.number_input("Number of causes to analyze:", min_value=2, max_value=20, value=5, key="pareto_num")
    
    causes = []
    impacts = []
    
    for i in range(int(num_causes)):
        col1, col2 = st.columns([2, 1])
        with col1:
            cause = st.text_input(f"Cause {i+1}:", value=pareto_data[i]['cause'] if i < len(pareto_data) else "", 
                                 key=f"pareto_cause_{i}")
        with col2:
            impact = st.number_input(f"Impact/Frequency:", min_value=0, value=pareto_data[i]['impact'] if i < len(pareto_data) else 10,
                                    key=f"pareto_impact_{i}")
        
        if cause:
            causes.append(cause)
            impacts.append(impact)
    
    # Update session state
    st.session_state.analysis_data['pareto'] = [
        {'cause': c, 'impact': i} for c, i in zip(causes, impacts)
    ]
    
    # Generate Pareto chart if we have data
    if causes and impacts:
        st.subheader("📈 Pareto Chart")
        
        # Sort by impact descending
        sorted_data = sorted(zip(causes, impacts), key=lambda x: x[1], reverse=True)
        sorted_causes, sorted_impacts = zip(*sorted_data) if sorted_data else ([], [])
        
        # Calculate cumulative percentage
        total = sum(sorted_impacts)
        cumulative = []
        cum_sum = 0
        for impact in sorted_impacts:
            cum_sum += impact
            cumulative.append(cum_sum / total * 100)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(sorted_causes),
            y=list(sorted_impacts),
            name='Impact',
            marker_color='#1f77b4',
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(sorted_causes),
            y=cumulative,
            name='Cumulative %',
            marker_color='#d62728',
            marker_size=8,
            yaxis='y2',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Pareto Chart',
            xaxis=dict(title='Causes'),
            yaxis=dict(title='Impact/Frequency', side='left'),
            yaxis2=dict(title='Cumulative Percentage', side='right', overlaying='y', range=[0, 100]),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight vital few
        st.markdown("### 💡 Key Insights")
        vital_few = []
        for i, (cause, cum_pct) in enumerate(zip(sorted_causes, cumulative)):
            if cum_pct <= 80:
                vital_few.append(cause)
        
        if vital_few:
            st.success(f"**Vital Few (causing ~80% of problem):** {', '.join(vital_few)}")
            st.info("Focus your efforts on addressing these causes first for maximum impact!")
    
    if st.button("✓ Complete Pareto Analysis"):
        st.session_state.completed = True
        st.rerun()

def render_matrix_analysis():
    """Render Root Cause Matrix or BCG Matrix"""
    st.subheader("📋 Prioritization Matrix")
    st.write(f"**Problem:** {st.session_state.problem_statement}")
    
    matrix_data = st.session_state.analysis_data.get('matrix', [])
    
    num_causes = st.number_input("Number of potential causes:", min_value=2, max_value=15, value=5, key="matrix_num")
    
    causes = []
    impacts = []
    efforts = []
    
    for i in range(int(num_causes)):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            cause = st.text_input(f"Cause {i+1}:", value=matrix_data[i]['cause'] if i < len(matrix_data) else "",
                                 key=f"matrix_cause_{i}")
        with col2:
            impact = st.slider(f"Impact (1-10):", min_value=1, max_value=10, 
                              value=matrix_data[i]['impact'] if i < len(matrix_data) else 5,
                              key=f"matrix_impact_{i}")
        with col3:
            effort = st.slider(f"Effort (1-10):", min_value=1, max_value=10,
                              value=matrix_data[i]['effort'] if i < len(matrix_data) else 5,
                              key=f"matrix_effort_{i}")
        
        if cause:
            causes.append(cause)
            impacts.append(impact)
            efforts.append(effort)
    
    st.session_state.analysis_data['matrix'] = [
        {'cause': c, 'impact': i, 'effort': e} for c, i, e in zip(causes, impacts, efforts)
    ]
    
    # Create scatter plot
    if causes:
        st.subheader("📊 Impact vs. Effort Matrix")
        
        df = pd.DataFrame({
            'Cause': causes,
            'Impact': impacts,
            'Effort': efforts
        })
        
        fig = px.scatter(df, x='Effort', y='Impact', text='Cause',
                        size_max=15, height=500,
                        color='Impact',
                        color_continuous_scale='RdYlGn')
        
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        
        # Add quadrant lines
        fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=2.5, y=7.5, text="Quick Wins<br>(High Impact, Low Effort)", 
                          showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=7.5, y=7.5, text="Major Projects<br>(High Impact, High Effort)", 
                          showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=2.5, y=2.5, text="Fill-ins<br>(Low Impact, Low Effort)", 
                          showarrow=False, font=dict(size=12, color="yellow"))
        fig.add_annotation(x=7.5, y=2.5, text="Thankless Tasks<br>(Low Impact, High Effort)", 
                          showarrow=False, font=dict(size=12, color="red"))
        
        fig.update_layout(
            xaxis=dict(range=[0, 11], title='Effort Required'),
            yaxis=dict(range=[0, 11], title='Potential Impact'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Strategic Recommendations")
        quick_wins = df[(df['Impact'] >= 5) & (df['Effort'] < 5)]
        major_projects = df[(df['Impact'] >= 5) & (df['Effort'] >= 5)]
        
        if not quick_wins.empty:
            st.success(f"**Quick Wins:** {', '.join(quick_wins['Cause'].tolist())}")
        if not major_projects.empty:
            st.info(f"**Major Projects:** {', '.join(major_projects['Cause'].tolist())}")
    
    if st.button("✓ Complete Matrix Analysis"):
        st.session_state.completed = True
        st.rerun()

def render_completion():
    """Render completion summary"""
    st.balloons()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("## 🎉 Analysis Complete!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("📋 Summary")
    st.write(f"**Problem:** {st.session_state.problem_statement}")
    st.write(f"**Framework Used:** {FRAMEWORKS[st.session_state.selected_framework]['name']}")
    st.write(f"**Firm:** {FRAMEWORKS[st.session_state.selected_framework]['firm']}")
    
    st.subheader("📊 Key Findings")
    
    if st.session_state.selected_framework == "Five Whys":
        st.write("Root Cause Chain:")
        for i in range(1, 6):
            if st.session_state.five_whys_answers.get(i):
                st.write(f"**Why {i}:** {st.session_state.five_whys_answers[i]}")
    
    elif st.session_state.selected_framework == "Pareto":
        if st.session_state.analysis_data.get('pareto'):
            st.write("Top causes identified through Pareto analysis")
    
    elif st.session_state.selected_framework in ["Root Cause Matrix", "BCG Growth Share"]:
        if st.session_state.analysis_data.get('matrix'):
            st.write("Prioritized causes based on impact and effort")
    
    st.subheader("🎯 Next Steps")
    st.markdown("""
    1. **Validate findings** with additional data
    2. **Develop action plan** with clear owners and timelines
    3. **Implement countermeasures** and monitor results
    4. **Document learnings** for future reference
    """)
    
    if st.button("🔄 Start New Analysis"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Export option
    st.subheader("💾 Export Results")
    
    results = {
        "problem": st.session_state.problem_statement,
        "framework": st.session_state.selected_framework,
        "timestamp": "Analysis completed"
    }
    
    st.download_button(
        label="Download Analysis Summary (JSON)",
        data=json.dumps(results, indent=2),
        file_name="rca_analysis.json",
        mime="application/json"
    )

def main():
    """Main application logic"""
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/search--v1.png", width=80)
        st.title("Navigation")
        
        if st.session_state.problem_statement:
            st.success(f"✅ Problem Defined")
            st.info(f"**Framework:** {st.session_state.selected_framework or 'Not yet selected'}")
            
            if st.button("🏠 Home"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.info("Describe your problem to begin")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **RCA Pro** uses professional frameworks from:
        - McKinsey & Company
        - Bain & Company
        - Boston Consulting Group
        
        Built for structured problem-solving.
        """)
    
    # Main content based on state
    if not st.session_state.problem_statement:
        render_homepage()
    elif st.session_state.completed:
        render_completion()
    else:
        # Render framework-specific interface
        framework = st.session_state.selected_framework
        
        if framework == "Five Whys":
            render_five_whys()
        elif framework in ["Issue Tree", "MECE", "Hypothesis Driven"]:
            render_issue_tree()
        elif framework == "Fishbone":
            render_fishbone()
        elif framework == "Pareto":
            render_pareto()
        elif framework in ["Root Cause Matrix", "BCG Growth Share"]:
            render_matrix_analysis()
        else:
            # Default to issue tree
            render_issue_tree()

if __name__ == "__main__":
    main()
