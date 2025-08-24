# Learnings from Debugging and Development

## `ModuleNotFoundError: No module named 'src.dashboard.cargo_statistics_tab'`

### Issue Resolved
- **Problem**: The application failed to launch on Streamlit Community Cloud with `ModuleNotFoundError: No module named 'src.dashboard.cargo_statistics_tab'` at line 36 of `app.py`, along with similar errors for `live_vessels_tab` and `ships_berths_tab`.
- **Root Cause**: The import paths in `app.py` were pointing to the old module locations. These tab modules had been moved to the `src/dashboard/tabs/` subdirectory, but some import statements in `app.py` were not updated to reflect the new directory structure.
- **Solution**: Updated the import paths in `app.py` to include the `tabs` subdirectory:
  - `from src.dashboard.live_vessels_tab import render_live_vessels_tab` → `from src.dashboard.tabs.live_vessels_tab import render_live_vessels_tab`
  - `from src.dashboard.ships_berths_tab import render_ships_berths_tab` → `from src.dashboard.tabs.ships_berths_tab import render_ships_berths_tab`
  - The `cargo_statistics_tab` import was already corrected in a previous fix.

### Key Learning
- **Module Reorganization Impact**: When refactoring code by moving modules to different directories, all import statements across the entire codebase must be systematically updated. Missing even one import can cause deployment failures.
- **Deployment Environment Sensitivity**: Import path errors that might not surface during local development (due to Python's module resolution behavior) become critical failures in cloud deployment environments like Streamlit Community Cloud.
- **Systematic Import Verification**: After any module reorganization, it's essential to search the entire codebase for import statements that reference the moved modules to ensure complete consistency.

## `NameError: name 'st' is not defined`

### Issue Resolved
- **Problem**: The application failed to launch on Streamlit Community Cloud, reporting a `NameError`.
- **Root Cause**: The file `hk_port_digital_twin/src/utils/visualization.py` used the `@st.cache_data` decorator without importing the `streamlit` library. The alias `st` was therefore not defined, leading to the `NameError`.
- **Solution**: Added `import streamlit as st` to the top of `hk_port_digital_twin/src/utils/visualization.py`.

### Key Learning
- **Dependency Awareness in Modular Code**: When refactoring or creating new modules (like `visualization.py`), it's crucial to ensure that all dependencies are explicitly imported within that module's scope. A file must be self-sufficient in its imports.
- **Local vs. Cloud Environments**: This error did not surface in the local development environment, likely because `streamlit` was imported in the main application file (`app.py`), and its scope might have been inadvertently shared. However, Streamlit Community Cloud has a stricter execution environment where each module's dependencies must be explicitly declared. This highlights the importance of testing in an environment that closely mirrors production.

## Scenario-Dependent Performance Analytics Implementation

### Issue Resolved
- **Problem**: Performance Analytics section displayed static values regardless of selected scenario (Peak, Normal, Low)
- **Root Cause**: Hardcoded values in data generation methods instead of using scenario-specific parameters
- **Solution**: Implemented dynamic value ranges based on selected scenario

### Changes Made

#### 1. Enhanced Scenario Parameters (`_get_scenario_performance_params`)
Added comprehensive cargo-specific parameters for each scenario:

**Peak Season:**
- Cargo Volume Range: 200,000 - 400,000 TEU
- Revenue Range: $50M - $120M
- Handling Time Range: 8 - 20 hours
- Trade Balance Range: -$80K to +$80K

**Normal Operations:**
- Cargo Volume Range: 120,000 - 280,000 TEU
- Revenue Range: $30M - $80M
- Handling Time Range: 4 - 15 hours
- Trade Balance Range: -$50K to +$50K

**Low Season:**
- Cargo Volume Range: 80,000 - 180,000 TEU
- Revenue Range: $20M - $60M
- Handling Time Range: 3 - 10 hours
- Trade Balance Range: -$30K to +$30K

#### 2. Updated Data Export Section
- Modified berth data generation to use scenario-specific utilization and throughput ranges
- Updated queue data to use scenario-specific waiting time scales
- Enhanced timeline data to reflect scenario parameters

#### 3. Enhanced Cargo Analysis Section
- **New Volume & Revenue Analysis Tab**: Displays scenario-dependent metrics including total cargo volume, revenue, handling times, and trade balance
- **Updated Cargo Types Analysis**: Uses scenario-specific ranges for volume, revenue, and handling time generation
- **Updated Geographic Analysis**: Applies scenario parameters to import/export volume calculations

#### 4. Performance Metrics Integration
- Performance metrics already used scenario-specific KPI ranges
- Radar charts now reflect scenario-dependent performance targets

### Technical Implementation Details

#### Key Methods Modified:
1. `_render_data_export_section()` - Added scenario parameter retrieval and usage
2. `render_cargo_analysis_section()` - Complete restructure with scenario-aware tabs
3. `_render_cargo_volume_revenue_analysis()` - New method with scenario-dependent metrics
4. `_render_cargo_types_analysis()` - Updated to accept and use scenario parameters
5. `_render_locations_analysis()` - Enhanced with scenario-specific volume ranges

#### Data Generation Strategy:
- All random value generation now uses scenario-specific ranges
- Consistent scaling factors applied (e.g., 0.1x to 0.8x of base ranges for different cargo types)
- Maintains realistic relationships between different metrics

### Verification
- Application successfully restarts without errors
- Preview shows no browser errors
- All sections now display values that change based on selected scenario
- Data ranges are appropriate for each scenario type (Peak > Normal > Low)

### Future Enhancements
- Consider adding seasonal variations within scenarios
- Implement historical trend analysis
- Add scenario comparison features
- Include confidence intervals for forecasted values