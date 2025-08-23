import streamlit as st
import pandas as pd
from hk_port_digital_twin.src.scenarios import list_available_scenarios

def render_scenario_analysis_tab():
    """
    Renders the Scenario Analysis & Comparison tab.
    """
    st.subheader("🎯 Scenario Analysis & Comparison")
    st.markdown("Compare different operational scenarios and their impact on port performance")

    # Scenario selection for comparison
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Select Scenarios to Compare**")
        available_scenarios = list_available_scenarios()
        scenario1 = st.selectbox("Scenario 1", available_scenarios, key="scenario1_select")
        scenario2 = st.selectbox("Scenario 2", available_scenarios, key="scenario2_select", index=1 if len(available_scenarios) > 1 else 0)

    with col2:
        st.write("**Comparison Parameters**")
        compare_metrics = st.multiselect(
            "Metrics to Compare",
            ["Ship Arrival Rate", "Processing Efficiency", "Berth Utilization", "Waiting Times"],
            default=["Ship Arrival Rate", "Processing Efficiency"]
        )

    if st.button("🔄 Run Scenario Comparison", type="primary"):
        with st.spinner("Running scenario comparison..."):
            # Simulate comparison logic
            comparison_data = {
                'Scenario': [scenario1, scenario2],
                'Ship Arrival Rate': [85.2, 92.1],
                'Processing Efficiency': [78.5, 82.3],
                'Berth Utilization': [72.1, 76.8],
                'Waiting Times': [2.3, 1.9]
            }
            
            st.session_state.scenario_comparison_data = comparison_data
            st.success("Scenario comparison completed!")

    # Display comparison results if available
    if hasattr(st.session_state, 'scenario_comparison_data'):
        st.subheader("📊 Comparison Results")
        
        comparison_df = pd.DataFrame(st.session_state.scenario_comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        if "Ship Arrival Rate" in compare_metrics:
            import plotly.express as px
            fig = px.bar(comparison_df, x='Scenario', y='Ship Arrival Rate', 
                       title="Ship Arrival Rate by Scenario")
            st.plotly_chart(fig, use_container_width=True)
        
        if "Processing Efficiency" in compare_metrics:
            import plotly.express as px
            fig = px.bar(comparison_df, x='Scenario', y='Processing Efficiency', 
                       title="Processing Efficiency by Scenario")
            st.plotly_chart(fig, use_container_width=True)