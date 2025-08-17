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