## Learnings from Debugging

This document outlines the key learnings from debugging sessions, focusing on the root causes of errors and their resolutions.

### Session 1: `AttributeError` and `NameError`

**1. `AttributeError: 'list' object has no attribute 'empty'`**

*   **Symptom:** The application crashed with an `AttributeError` when checking if `berth_data` was empty.
*   **Root Cause:** The code was using `.empty`, a pandas DataFrame attribute, on a list object. This happened because the `load_data` function, which was temporarily removed, returned a list, not a DataFrame.
*   **Resolution:** The condition was changed from `if not berth_data.empty:` to `if berth_data:`, which is the correct way to check if a list is empty in Python.

**2. `NameError: name 'get_berth_config' is not defined`**

*   **Symptom:** The application failed to load real-time berth data due to a `NameError`.
*   **Root Cause:** The function `get_berth_config` was called, but it was not defined anywhere in the codebase. The correct function to load berth configurations is `load_berth_configurations`.
*   **Resolution:** The incorrect function call was replaced with `load_berth_configurations`, and the function was imported from `src/utils/data_loader.py`.

### Session 3: Scenario Update Regression

**1. `Regression: Dashboard not updating on scenario change`**

*   **Symptom:** The dashboard data did not update when the user changed the scenario (e.g., from 'Normal' to 'Peak') in the sidebar.
*   **Root Cause:** The `main` function in `streamlit_app.py` was loading the data only once when the application started. It did not reload the data when the scenario was changed via the sidebar controls.
*   **Resolution:** The `main` function was refactored to:
    1.  Get the current scenario from the `scenario_manager` at the beginning of each run.
    2.  Call the `load_data` function with the current scenario to ensure the data is always up-to-date.
    3.  Remove redundant calls to `initialize_session_state`, `create_sidebar`, and `load_data` to streamline the application flow and prevent conflicts.

**1. `TypeError: get_real_berth_data() takes 0 positional arguments but 1 was given`**

*   **Symptom:** The application failed to load real-time berth data, showing a `TypeError`.
*   **Root Cause:** The function `get_real_berth_data` was called with `berth_config` as an argument, but the function was defined to take no arguments.
*   **Resolution:** The function definition was updated to accept the `berth_config` argument, aligning the function signature with its usage.

**2. `TypeError: tuple indices must be integers or slices, not str`**

*   **Symptom:** The application crashed when trying to access data from a tuple as if it were a dictionary or DataFrame.
*   **Root Cause:** The `get_real_berth_data` function returns a tuple containing a DataFrame and a dictionary (`berths_df`, `berth_metrics`). The calling code was assigning this entire tuple to a single variable and then attempting to access it like a DataFrame.
*   **Resolution:** The return value of `get_real_berth_data` was unpacked into two separate variables (`berth_details`, `_`), and only the DataFrame was used.

**3. `ValueError: The truth value of a DataFrame is ambiguous`**

*   **Symptom:** The application crashed with a `ValueError` when checking if a DataFrame was empty.
*   **Root Cause:** The code was using `if berth_data:`, which is not a valid way to check if a pandas DataFrame is empty.
*   **Resolution:** The condition was changed to `if not berth_data.empty:`, which is the correct way to check for an empty DataFrame.

### Session 4: Dynamic Data Generation and Key Errors

**1. `KeyError: 'queue'`**

*   **Symptom:** The application crashed with a `KeyError` when trying to access `data['queue']` in the main function.
*   **Root Cause:** The `load_sample_data` function was modified to implement dynamic berth data generation but inadvertently stopped returning the 'queue' key in the data dictionary.
*   **Resolution:** The `load_sample_data` function was corrected to ensure it always returns all required keys including 'queue', 'berths', 'vessels', etc.

**2. `AttributeError: 'NoneType' object has no attribute 'empty'`**

*   **Symptom:** The application crashed when trying to check if `berth_data` was empty, but `berth_data` was `None`.
*   **Root Cause:** The main function was trying to access `data.get('berth_details')` but the data dictionary contained the berth information under the key 'berths', not 'berth_details'.
*   **Resolution:** Changed `data.get('berth_details')` to `data.get('berths')` to correctly access the berth data.

**3. Dynamic Scenario-Based Data Generation Implementation**

*   **Challenge:** Implement dynamic data generation that responds to scenario changes (Peak, Normal, Low) with different utilization rates and berth occupancy patterns.
*   **Solution:** 
    1. Defined scenario parameters with specific utilization and occupied berth ranges for each scenario
    2. Modified `load_sample_data` to use these parameters for generating randomized but realistic data
    3. Ensured the `load_data` function properly integrates with the scenario manager
    4. Fixed data flow issues to ensure scenario changes trigger proper data reloads

**Key Learnings:**
- Always ensure data dictionary keys are consistent between generation and consumption functions
- When modifying data generation logic, verify all return values and data structures remain intact
- Test scenario switching functionality thoroughly to ensure dynamic updates work correctly
- Use proper error handling and validation when accessing dictionary keys that might not exist