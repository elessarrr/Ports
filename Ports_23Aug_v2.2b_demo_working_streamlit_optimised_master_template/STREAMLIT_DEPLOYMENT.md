# Streamlit Community Cloud Deployment Guide

## Overview
This guide explains how to deploy the Hong Kong Port Digital Twin Dashboard to Streamlit Community Cloud.

## Files Created for Deployment

### 1. `streamlit_app.py` (Root Level Entry Point)
This file serves as the main entry point for Streamlit Community Cloud deployment. Streamlit Community Cloud expects the main app file to be named `streamlit_app.py` at the project root level.

**Key Features:**
- Sets up Python path to include the `hk_port_digital_twin` directory
- Changes working directory for proper relative imports
- Imports and runs the actual dashboard from `hk_port_digital_twin/src/dashboard/streamlit_app.py`
- Includes error handling for missing dependencies or import issues

### 2. `requirements.txt` (Root Level Dependencies)
Contains all necessary Python packages for the dashboard:
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `watchdog` - File monitoring
- `requests` - HTTP requests
- `schedule` - Task scheduling
- `python-dotenv` - Environment variables
- `scipy` - Scientific computing
- `simpy>=4.0.0` - Discrete event simulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `statsmodels` - Statistical modeling

## Deployment Steps

### 1. Prepare Your Repository
1. Ensure your code is pushed to a GitHub repository
2. Verify that both `streamlit_app.py` and `requirements.txt` are at the root level
3. Make sure all your project files are in the `hk_port_digital_twin/` directory

### 2. Deploy to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `streamlit_app.py` (should be auto-detected)
6. Click "Deploy!"

### 3. Monitor Deployment
- The deployment process will install dependencies from `requirements.txt`
- Watch the logs for any errors during installation or startup
- The app should be accessible via the provided Streamlit Community Cloud URL

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are listed in `requirements.txt`
2. **Path Issues**: The entry point handles path setup automatically
3. **Data Files**: Make sure all required data files are included in your repository

### Local Testing
Before deploying, test the entry point locally:
```bash
streamlit run streamlit_app.py
```

## Project Structure
```
project-root/
├── streamlit_app.py          # Entry point for Streamlit Community Cloud
├── requirements.txt          # Dependencies
├── STREAMLIT_DEPLOYMENT.md   # This guide
└── hk_port_digital_twin/     # Main project directory
    ├── src/
    │   └── dashboard/
    │       └── streamlit_app.py  # Actual dashboard application
    ├── config/
    ├── raw_data/
    └── requirements.txt      # Original project dependencies
```

## Notes
- The original `run_demo.py` launcher is not needed for Streamlit Community Cloud
- All relative imports and file paths are handled by the entry point
- The dashboard retains all its original functionality when deployed
