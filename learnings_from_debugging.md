# Learnings from Debugging

This document outlines the debugging process for resolving two critical errors in the Hong Kong Port Digital Twin Dashboard.

## Error 1: `AttributeError: 'list' object has no attribute 'empty'`

### Symptom

The Streamlit application crashed with the following error:

```
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

### Root Cause

The `get_berth_config` function was called in `streamlit_app.py` to load berth configurations, but the function did not exist. The correct function for this purpose was `load_berth_configurations`, which is available in `src/utils/data_loader.py`.

### Resolution

The resolution involved two steps:

1.  **Correcting the function call:** The call to `get_berth_config()` was replaced with `load_berth_configurations()`.
2.  **Importing the function:** The `load_berth_configurations` function was already being imported from `hk_port_digital_twin.src.utils.data_loader` at the top of `streamlit_app.py`, so no new import was needed.

By making these changes, the application was able to correctly load the berth configurations, and the dashboard's functionality was restored.