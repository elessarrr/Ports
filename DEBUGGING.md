# Debugging `AttributeError: st.session_state has no attribute "scenario"`

## 1. Error Description

The application was crashing with the following error:

```
AttributeError: st.session_state has no attribute "scenario". Did you forget to initialize it? More info: https://docs.streamlit.io/develop/concepts/architecture/session-state#initialization
Traceback:
File "/Users/Bhavesh/opt/anaconda3/lib/python3.8/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
result = func()
File "/Users/Bhavesh/opt/anaconda3/lib/python3.8/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec
exec(code, module.__dict__)
File "/Users/Bhavesh/Documents/GitHub/Ports/Ports/hk_port_digital_twin/src/dashboard/streamlit_app.py", line 2360, in <module>
main()
File "/Users/Bhavesh/Documents/GitHub/Ports/Ports/hk_port_digital_twin/src/dashboard/streamlit_app.py", line 455, in main
data = load_sample_data(st.session_state.scenario)
File "/Users/Bhavesh/opt/anaconda3/lib/python3.8/site-packages/streamlit/runtime/state/session_state_proxy.py", line 131, in __getattr__
raise AttributeError(_missing_attr_error_message(key))
```

This error occurred because the `st.session_state.scenario` attribute was being accessed before it was initialized.

## 2. Root Cause Analysis

The traceback indicates that the error originates in the `main` function of `streamlit_app.py` at line 455, where `load_sample_data(st.session_state.scenario)` is called.

The `st.session_state` object in Streamlit is used to store session-specific data. However, attributes of `st.session_state` must be initialized before they can be accessed. In this case, `st.session_state.scenario` was not initialized, leading to the `AttributeError`.

The call to `load_sample_data` happens in the `main` function. The `initialize_session_state` function is called at the beginning of `main`, which is the correct place to initialize session state variables. However, the initialization for `scenario` was missing.

## 3. Solution

The solution was to initialize `st.session_state.scenario` with a default value in the `initialize_session_state` function. This ensures that the attribute exists before it is accessed later in the application.

The following code was added to the `initialize_session_state` function in `hk_port_digital_twin/src/dashboard/streamlit_app.py`:

```python
if 'scenario' not in st.session_state:
    st.session_state.scenario = 'normal'
```

This change ensures that `st.session_state.scenario` is always available, preventing the `AttributeError`.

## 4. Verification

The fix was verified by running the application again. The `AttributeError` no longer occurs, and the application now loads with the default 'normal' scenario as expected. The dashboard now correctly displays the data for the selected scenario, and the user can switch between scenarios without any issues.