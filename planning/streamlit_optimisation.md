# Streamlit Application Optimisation Plan

This document outlines a plan to address performance issues in the Streamlit application, focusing on creating a more responsive and stable user experience.

## 1. Disable Real-Time Updates for Deployment

**Problem:** The `RealTimeDataManager` runs in a separate thread, fetching data every 15 seconds. This is a likely cause of deployment instability and performance bottlenecks.

**Solution:**
- **Short-term:** For the initial deployment and performance testing, we will disable the real-time data fetching feature. This can be done by commenting out the initialisation and execution of the `RealTimeDataManager`.
- **Long-term:** Investigate more robust methods for handling background tasks in a Streamlit environment, such as using a separate microservice or a more sophisticated caching strategy.

## 2. Implement Caching for Data Loading

**Problem:** Several data loading functions, such as `load_combined_vessel_data`, are computationally expensive and are re-executed on every interaction, leading to slow load times.

**Solution:**
- Apply Streamlit's caching decorators (`@st.cache_data` or `@st.cache_resource`) to all heavy data loading and processing functions. This will ensure that these operations are only performed once, and the results are stored in memory for subsequent use.
- **Priority Functions for Caching:**
    - `load_combined_vessel_data`
    - `load_cargo_data`
    - `get_real_berth_data` (with an appropriate `ttl` to refresh periodically if needed)

## 3. Optimise Visualisation Rendering

**Problem:** The dashboard renders numerous complex Plotly charts on initial load, which can be resource-intensive and contribute to slow rendering times.

**Solution:**
- **Defer Complex Chart Loading:** Instead of rendering all charts at once, we will implement a mechanism to load them on demand. For example, we can display a button or a placeholder, and the chart will only be generated and displayed when the user clicks on it.
- **Simplify Initial Views:** Provide simplified summary visualisations on the initial dashboard load. More detailed and complex charts can be made available through user interaction.

## 4. General Code Review and Refactoring

**Problem:** There may be other areas in the code with inefficient data processing, redundant computations, or suboptimal logic.

**Solution:**
- Conduct a thorough code review of `streamlit_app.py` to identify any further opportunities for optimisation.
- **Areas of Focus:**
    - **Data Filtering:** Ensure that data filtering operations are performed efficiently.
    - **Looping Constructs:** Review loops to ensure they are not performing unnecessary computations.
    - **Session State Management:** Optimise the use of `st.session_state` to avoid storing large objects that could slow down the application.