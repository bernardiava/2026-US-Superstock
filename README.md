# 🔍 RCA Pro - Interactive Root Cause Analysis App

A professional-grade Streamlit application for root cause analysis using frameworks from McKinsey, Bain, and BCG.

## Features

### 🎯 Auto-Detection of Frameworks
The app automatically detects the best framework based on your problem description:
- **Five Whys** - For simple problems and process issues
- **Issue Tree** - For complex strategic problems
- **MECE Principle** - For comprehensive categorization
- **Hypothesis-Driven** - For time-constrained projects
- **Fishbone (Ishikawa)** - For manufacturing and quality issues
- **Pareto Analysis** - For data-rich prioritization
- **Root Cause Matrix** - For multi-factorial problems
- **BCG Matrix Adaptation** - For portfolio prioritization

### 📊 Interactive Visualizations
- Dynamic Issue Tree diagrams
- Pareto charts with cumulative percentages
- Impact vs. Effort matrices
- Fishbone category analysis

### 💼 Professional Frameworks
Frameworks used by top consulting firms:
- **McKinsey & Company**: MECE, Issue Trees, Hypothesis-Driven
- **Bain & Company**: Five Whys, Root Cause Matrix, Fishbone
- **Boston Consulting Group**: BCG Matrix, Pareto Analysis

## Installation

```bash
pip install streamlit pandas plotly
```

## Usage

```bash
streamlit run rca_app.py
```

Then open your browser to `http://localhost:8501`

## How It Works

1. **Describe Your Problem**: Enter a detailed description of the issue you're analyzing
2. **Auto-Detection**: The app analyzes keywords and suggests the best framework
3. **Step-by-Step Guidance**: Follow the interactive workflow for your selected framework
4. **Visualization**: View dynamic charts and diagrams of your analysis
5. **Export Results**: Download your analysis as JSON for documentation

## Example Problems

- "Customer satisfaction dropped 15% in Q3 due to long wait times" → Pareto Analysis
- "Manufacturing defect rate increased in production line 3" → Fishbone Diagram
- "Why did the server outage occur?" → Five Whys
- "How to improve market share in Asia-Pacific region" → Issue Tree

## Deployment

### Local Development
```bash
streamlit run rca_app.py --server.port 8501
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repository to [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY rca_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "rca_app.py"]
```

## Requirements

- Python 3.8+
- streamlit >= 1.57.0
- pandas >= 2.0.0
- plotly >= 6.0.0

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
