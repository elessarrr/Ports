# Learnings from Debugging

This document outlines the debugging process for resolving two critical errors in the Hong Kong Port Digital Twin Dashboard.

## Error 1: `AttributeError: 'list' object has no attribute 'empty'`

### Symptom

The Streamlit application crashed with the following error:

```

## Error 3: `NameError: name 'UnifiedSimulationFramework' is not defined`

### Symptom

The Streamlit application failed to start with the following error:

```
NameError: name 'UnifiedSimulationFramework' is not defined
Traceback:
File "/Users/Bhavesh/Documents/GitHub/Ports/Ports/hk_port_digital_twin/src/dashboard/streamlit_app.py", line 1760, in <module>
    main()
File "/Users/Bhavesh/Documents/GitHub/Ports/Ports/hk_port_digital_twin/src/dashboard/streamlit_app.py", line 1752, in main
    st.session_state.unified_simulations_tab = UnifiedSimulationsTab()
File "/Users/Bhavesh/Documents/GitHub/Ports/Ports/hk_port_digital_twin/src/dashboard/unified_simulations_tab.py", line 84, in __init__
    self.unified_framework = UnifiedSimulationFramework()
```

### Root Cause

The error was caused by an incorrect class name reference in `unified_simulations_tab.py`. The code was trying to instantiate `UnifiedSimulationFramework()`, but this class does not exist. The correct class name is `UnifiedSimulationController`, which was properly imported but not used correctly.

### Resolution

The fix involved correcting the class name reference in the initialization.

**Incorrect Code:**

```python
self.unified_framework = UnifiedSimulationFramework()
```

**Corrected Code:**

```python
self.unified_framework = UnifiedSimulationController()
```

### Key Learnings

1. **Import vs Usage Consistency**: Even when the correct class is imported, ensure the instantiation uses the correct class name.
2. **Class Naming Conventions**: During refactoring, verify that all references to renamed classes are updated consistently across the codebase.
3. **Error Tracing**: The error traceback clearly pointed to the exact line and file where the undefined name was being used, making diagnosis straightforward.
AttributeError: 'list' object has no attribute 'empty'
```

This error occurred in `streamlit_app.py` at line 1494, within the "Live Berths" tab (`tab7`).

### Root Cause

The error was caused by an incorrect attempt to check if a list was empty using the `.empty` attribute, which is a feature of pandas DataFrames, not standard Python lists. The `berth_data` variable, which was being checked, was a list of dictionaries, not a DataFrame.

### Resolution

The fix involved changing the conditional check to correctly determine if the list is empty.

**Incorrect Code:**

```python
if not berth_data.empty:
```

**Corrected Code:**

```python
if berth_data:
```

This change correctly checks for the presence of elements in the `berth_data` list.

## Error 2: `NameError: name 'get_berth_config' is not defined`

### Symptom

After resolving the `AttributeError`, the dashboard displayed a new error:

```
Could not load real-time berth data: name 'get_berth_config' is not defined
```

This error indicated that the `get_berth_config` function was being called but was not defined or imported in the current scope.

## Error 3: `ImportError: cannot import name 'load_combined_vessel_data'`

### Symptom

The Streamlit application failed to start with the following import error:

```
ImportError: cannot import name 'load_combined_vessel_data' from 'hk_port_digital_twin.src.utils.data_loader'
```

This error occurred in `streamlit_app.py` at line 16, during the import statement.

### Root Cause Analysis

Initial investigation revealed:
1. The `load_combined_vessel_data` function was properly defined in `data_loader.py` at line 731
2. The function had correct syntax and proper indentation
3. All dependencies (`load_arriving_ships`, `load_vessel_arrivals`) were also properly defined
4. Direct import testing via command line worked successfully

### Resolution

The issue was resolved by restarting the Streamlit application. This suggests the error was likely caused by:
- A temporary Python module caching issue
- The Streamlit development server not detecting recent changes to the `data_loader.py` file
- A race condition during the previous application startup

**Actions Taken:**
1. Verified function existence and syntax in `data_loader.py`
2. Tested direct import via command line (successful)
3. Stopped the running Streamlit process
4. Restarted the Streamlit application
5. Confirmed successful loading with proper vessel data integration

### Key Learnings

- Streamlit's hot-reload mechanism may not always detect changes to imported modules
- When encountering import errors for recently added functions, try restarting the development server
- Always verify function definitions exist before assuming syntax or dependency issues
- Use direct Python import testing to isolate whether the issue is with the module or the application server
-6. **Data Deduplication Logic**: When combining datasets with overlapping records, consider status priority rather than simple "first wins" deduplication
7. **Status Value Debugging**: Always verify actual data values in combined datasets, especially when UI metrics don't match expectations

---

## Issue 3: Missing Arriving Ships in Vessel Table

### Symptom
- No ships showing as "arriving" in the Live Vessel Arrivals tab
- Metrics showing 0 arriving vessels despite data being loaded
- Only "in_port" and "departed" statuses visible

### Root Cause Analysis
1. **Data Source Overlap**: Both `load_vessel_arrivals()` and `load_arriving_ships()` were loading from the same XML file (`Arrived_in_last_36_hours.xml`)
2. **Status Assignment Difference**: 
   - `load_vessel_arrivals()` assigns status as 'in_port' or 'departed'
   - `load_arriving_ships()` assigns status as 'arriving' or 'departed'
3. **Deduplication Logic**: The `drop_duplicates(keep='first')` was keeping the first occurrence (from vessel_arrivals), losing the 'arriving' status from the second dataset
4. **UI Function Mismatch**: Tab4 was initially using `load_vessel_arrivals()` instead of `load_combined_vessel_data()`

### Resolution
1. **Updated Tab4**: Changed from `load_vessel_arrivals()` to `load_combined_vessel_data()`
2. **Improved Deduplication**: Implemented status priority logic:
   ```python
   # Priority: arriving > in_port > departed
   status_priority = {'arriving': 3, 'in_port': 2, 'departed': 1}
   combined_df['priority'] = combined_df['status'].map(status_priority)
   combined_df = combined_df.sort_values('priority', ascending=False)
   combined_df = combined_df.drop_duplicates(subset=['vessel_name', 'call_sign'], keep='first')
   ```
3. **Updated UI Metrics**: Changed metrics to show "Total Vessels", "Arriving", "In Port", "Departed"
4. **Column Name Handling**: Added compatibility for different column names between datasets
5. **Fixed Vessel Charts**: Updated `render_arriving_ships_list()` in `vessel_charts.py` to:
   - Use `load_combined_vessel_data()` instead of `load_vessel_arrivals()`
   - Filter for `status == 'arriving'` instead of `status == 'in_port'`
   - Updated display messages and section titles

### Key Learnings
1. **Data Source Analysis**: Always verify what data sources are being used by different loading functions
2. **Deduplication Strategy**: Consider business logic priority when merging overlapping datasets
3. **Status Consistency**: Ensure UI functions use the correct data loading methods
4. **Debugging Approach**: Use targeted debug scripts to isolate data loading vs UI display issues
5. **UI Component Consistency**: Check all UI components that display vessel data to ensure they use consistent data sources and filters

### Result
- 34 vessels now correctly show as "arriving"
- 59 vessels show as "departed" 
- Total: 93 vessels
- Data sources: 77 from 'arriving_ships', 16 from 'current_arrivals'
- All UI components now consistently display arriving ships

```### Root Cause

The `get_berth_config` function was called in `streamlit_app.py` to load berth configurations, but the function did not exist. The correct function for this purpose was `load_berth_configurations`, which is available in `src/utils/data_loader.py`.

### Resolution

The resolution involved two steps:

1.  **Correcting the function call:** The call to `get_berth_config()` was replaced with `load_berth_configurations()`.
2.  **Importing the function:** The `load_berth_configurations` function was already being imported from `hk_port_digital_twin.src.utils.data_loader` at the top of `streamlit_app.py`, so no new import was needed.

By making these changes, the application was able to correctly load the berth configurations, and the dashboard's functionality was restored.

## Error 3: `DataFrame.sort_values() got an unexpected keyword argument 'na_last'`

### Symptom

The arriving ships list functionality failed with the following error:

```
Error loading arriving ships data: DataFrame.sort_values() got an unexpected keyword argument 'na_last'
```

This error occurred in the `render_arriving_ships_list()` function in `vessel_charts.py` when attempting to sort vessel data by arrival time.

### Root Cause

The error was caused by using an incorrect parameter name in the pandas `sort_values()` method. The code used `na_last=True`, but the correct parameter name is `na_position='last'`. This inconsistency occurred because different parts of the codebase were using different parameter names for handling null values during sorting.

### Resolution

The fix involved changing the parameter name from `na_last=True` to `na_position='last'` to match the correct pandas API.

**Incorrect Code:**

```python
arriving_ships = arriving_ships.sort_values('arrival_time', ascending=False, na_last=True)
```

**Corrected Code:**

```python
arriving_ships = arriving_ships.sort_values('arrival_time', ascending=False, na_position='last')
```

### Prevention

1. **Consistent Parameter Usage**: Ensure all pandas operations use consistent parameter names throughout the codebase
2. **Code Review**: Review existing code patterns before implementing new functionality
3. **Documentation Reference**: Always refer to the official pandas documentation for correct parameter names
4. **Testing**: Test new functionality thoroughly to catch parameter errors early

## How to document

When documenting debugging insights, follow this format:

1. **Error Description**: Brief description of the error
2. **Symptom**: What was observed (error messages, unexpected behavior)
3. **Root Cause**: The underlying issue that caused the problem
4. **Resolution**: The specific fix applied
5. **Prevention**: How to avoid this issue in the future
6. **Code Examples**: Before and after code snippets when relevant