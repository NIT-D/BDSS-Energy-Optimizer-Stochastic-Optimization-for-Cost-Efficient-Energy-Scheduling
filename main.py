# main.py
import streamlit as st
import pandas as pd
import numpy as np
from solving import build_stochastic_model, solve_model, extract_solution
from plotting import (
    plot_costs_plotly,
    plot_scenario_adjustments_plotly,
    plot_base_schedule_plotly
)
import plotly.graph_objects as go

# ------------------------
# PAGE CONFIG & STYLING
# ------------------------
st.set_page_config(
    page_title="⚡ BDSS Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .block-container {padding-top: 1rem;}
    .stMetric {background-color: #262730; padding: 0.8rem; border-radius: 0.5rem;}
    h1, h2, h3 {color: #00FF88;}
</style>
""", unsafe_allow_html=True)

# ------------------------
# HEADER
# ------------------------
st.title("⚡ BDSS Energy Optimizer")
st.caption("Interactive Decision Support System for Cost-Efficient Energy Scheduling under Price Uncertainty")

# ------------------------
# SIDEBAR - FILE UPLOAD
# ------------------------
st.sidebar.header("📂 Upload Electricity Price Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file with 'Prices' and 'Probability' sheets",
    type=["xlsx"],
    key="price_file_upload"
)

if uploaded_file:
    try:
        with st.spinner("📊 Loading and validating data..."):
            xls = pd.ExcelFile(uploaded_file)
            price_sheet = next((s for s in xls.sheet_names if s.lower() == "prices"), None)
            prob_sheet = next((s for s in xls.sheet_names if s.lower() == "probability"), None)

            if not price_sheet or not prob_sheet:
                st.error("❌ Required sheets not found.")
                st.stop()

            df_prices = xls.parse(price_sheet).drop(columns=["Unnamed: 0"], errors="ignore")
            df_prob = xls.parse(prob_sheet)

            scenario_columns = [col for col in df_prices.columns if "Scenario" in col]
            if not scenario_columns:
                st.error("❌ No scenario columns found.")
                st.stop()

            if len(df_prob.columns) != len(scenario_columns):
                st.error(f"❌ Mismatch: {len(scenario_columns)} scenario columns but {len(df_prob.columns)} probabilities.")
                st.stop()

            prob_sum = df_prob.iloc[0].sum()
            if abs(prob_sum - 1.0) > 0.01:
                st.warning(f"⚠️ Probabilities sum to {prob_sum:.3f}, normalizing...")
                df_prob = df_prob / prob_sum

            st.success(f"✅ Data loaded: {len(df_prices)} rows, {len(scenario_columns)} scenarios")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # ------------------------
    # FILTER DATA
    # ------------------------
    df_prices['Timestamp'] = pd.date_range(start="2024-01-01", periods=len(df_prices), freq='h')
    df_prices['DayOfWeek'] = df_prices['Timestamp'].dt.day_name()
    df_prices['Hour'] = df_prices['Timestamp'].dt.hour

    def get_season(date):
        if date.month in [12, 1, 2]:
            return 'Winter'
        elif date.month in [3, 4, 5]:
            return 'Spring'
        elif date.month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df_prices['Season'] = df_prices['Timestamp'].apply(get_season)

    season_selected = st.sidebar.selectbox("🌤️ Select Season", df_prices['Season'].unique())
    day_selected = st.sidebar.selectbox("📅 Select Day of Week", df_prices['DayOfWeek'].unique())

    filtered_df = df_prices[(df_prices['Season'] == season_selected) & (df_prices['DayOfWeek'] == day_selected)]
    if len(filtered_df) < 24:
        st.error("Not enough rows for selection.")
        st.stop()

    # ------------------------
    # AVERAGE DAY PROFILE
    # ------------------------
    scenario_columns = [col for col in filtered_df.columns if "Scenario" in col]
    average_day_df = (
        filtered_df
        .groupby('Hour')[scenario_columns]
        .mean()
        .reset_index()
        .sort_values('Hour')
        .reset_index(drop=True)
    )

    # €/MWh → €/kWh
    for col in scenario_columns:
        average_day_df[col] = average_day_df[col] / 1000.0

    # ------------------------
    # SIDEBAR - STRATEGY & PARAMETERS
    # ------------------------
    st.sidebar.header("⚙️ Optimization Strategy & Parameters")
    preset = st.sidebar.selectbox(
        "🎯 Choose Optimization Strategy:",
        ["Custom", "Cost Minimization", "Smooth Operation", "Risk Averse", "Balanced"],
        help="Select a preset strategy to auto-fill parameters, then edit as needed"
    )

    if preset == "Cost Minimization":
        default_total_demand, default_x_min, default_x_max = 100, 0.0, 15.0
        default_penalty, default_max_ramp = 0.1, 5.0
        default_risk_weight, default_robust_weight, default_flexibility = 0.0, 0.0, 0.0
        default_base_cov, default_price_sens, default_cvar_alpha = 0.80, 0.10, 0.90
    elif preset == "Smooth Operation":
        default_total_demand, default_x_min, default_x_max = 100, 2.0, 8.0
        default_penalty, default_max_ramp = 5.0, 1.0
        default_risk_weight, default_robust_weight, default_flexibility = 0.05, 0.02, 0.2
        default_base_cov, default_price_sens, default_cvar_alpha = 0.90, 0.20, 0.95
    elif preset == "Risk Averse":
        default_total_demand, default_x_min, default_x_max = 100, 1.0, 12.0
        default_penalty, default_max_ramp = 2.0, 2.5
        default_risk_weight, default_robust_weight, default_flexibility = 0.30, 0.20, 0.8
        default_base_cov, default_price_sens, default_cvar_alpha = 0.95, 0.30, 0.97
    elif preset == "Balanced":
        default_total_demand, default_x_min, default_x_max = 100, 1.0, 10.0
        default_penalty, default_max_ramp = 1.0, 2.0
        default_risk_weight, default_robust_weight, default_flexibility = 0.10, 0.05, 0.5
        default_base_cov, default_price_sens, default_cvar_alpha = 0.90, 0.30, 0.95
    else:  # Custom defaults
        default_total_demand, default_x_min, default_x_max = 100, 0.0, 10.0
        default_penalty, default_max_ramp = 1.0, 2.0
        default_risk_weight, default_robust_weight, default_flexibility = 0.10, 0.05, 0.5
        default_base_cov, default_price_sens, default_cvar_alpha = 0.90, 0.30, 0.95

    # ------------------------
    # PARAMETER SLIDERS
    # ------------------------
    st.sidebar.markdown("**📊 Basic Parameters:**")
    total_demand = st.sidebar.slider("Total Demand (kWh)", 50, 500, default_total_demand, 10)
    x_min = st.sidebar.slider("Min Usage (kWh)", 0.0, 10.0, default_x_min, 0.5)
    x_max = st.sidebar.slider("Max Usage (kWh)", 1.0, 20.0, default_x_max, 0.5)

    st.sidebar.markdown("**🔧 Operational Constraints:**")
    penalty = st.sidebar.slider("Smoothness Penalty", 0.0, 10.0, default_penalty, 0.1)
    max_ramp = st.sidebar.slider("Max Ramp (kWh)", 0.5, 5.0, default_max_ramp, 0.1)

    st.sidebar.markdown("**🎲 Risk Management:**")
    risk_weight = st.sidebar.slider("Risk Aversion (CVaR weight)", 0.0, 1.0, default_risk_weight, 0.05)
    robust_weight = st.sidebar.slider("Robustness (Worst-case weight)", 0.0, 1.0, default_robust_weight, 0.01)
    real_time_flexibility = st.sidebar.slider("Real-time Flexibility (±kWh)", 0.0, 2.0, default_flexibility, 0.1)

    st.sidebar.markdown("**🧠 Advanced (Coverage & Sensitivity):**")
    base_coverage_target = st.sidebar.slider("Min Base Coverage (fraction of demand)", 0.0, 1.0, default_base_cov, 0.05)
    price_sensitivity = st.sidebar.slider("Price Sensitivity (Δdemand for 100% price change)", 0.0, 1.0, default_price_sens, 0.05)
    cvar_alpha = st.sidebar.slider("CVaR Confidence (α)", 0.80, 0.99, default_cvar_alpha, 0.01)

    # ------------------------
    # MODEL DATA PREP
    # ------------------------
    p_base = average_day_df[scenario_columns].mean(axis=1).tolist()
    scenarios = [average_day_df[col].tolist() for col in scenario_columns]
    probabilities = {i: float(df_prob.iloc[0, i]) for i in range(len(df_prob.columns))}

    # ------------------------
    # BUILD & SOLVE MODEL
    # ------------------------
    model = build_stochastic_model(
        p_base, scenarios, probabilities,
        total_demand, x_min, x_max,
        penalty, max_ramp, risk_weight, robust_weight, real_time_flexibility,
        base_coverage_target, price_sensitivity, cvar_alpha
    )

    with st.spinner("⚙️ Solving optimization model..."):
        results = solve_model(model, solver_name="gurobi")
        if str(results.solver.termination_condition) != "optimal":
            st.error("❌ Solver failed.")
            st.stop()

    solution = extract_solution(model)

    # ------------------------
    # METRICS + RESULTS SUMMARY
    # ------------------------
    usage_values = list(solution['x'].values())
    total_usage = sum(usage_values)
    base_coverage = (total_usage / total_demand) * 100
    scenario_costs_for_display = {s: solution['scenario_costs'][s] for s in solution['scenario_costs']}
    expected_cost = sum(probabilities[s] * scenario_costs_for_display[s] for s in range(len(scenarios)))

    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Expected Cost", f"€{expected_cost:.2f}")
    col2.metric("⚡ Base Usage", f"{total_usage:.1f} kWh")
    col3.metric("📈 Peak Usage", f"{max(usage_values):.2f} kWh")
    col4.metric("📊 Usage Variance", f"{np.var(usage_values):.3f}")
    col5.metric("📌 Base Coverage", f"{base_coverage:.1f}%")

    # ------------------------
    # TABS FOR VISUALS + RISK + SUMMARY
    # ------------------------
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Schedule", "🎲 Scenarios", "💰 Costs", "🎯 Risk Analysis", "📋 Optimization Summary"]
    )

    with tab1:
        # Base schedule with price overlay
        plot_base_schedule_plotly(solution['x'], p_base)

        # Scenario playback
        scenario_names = [f"Scenario {i+1}" for i in range(len(scenarios))]
        selected_scenario_name = st.selectbox(
            "🎯 Select Scenario to Visualize Adjusted Plan", scenario_names, key="scenario_playback"
        )
        selected_scenario_index = scenario_names.index(selected_scenario_name)

        base_schedule = solution['x']
        scenario_adjustments = solution['x_s'][selected_scenario_index]
        adjusted_schedule = {h: base_schedule[h] + scenario_adjustments[h] for h in base_schedule}
        price_curve = scenarios[selected_scenario_index]  # €/kWh

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(base_schedule.keys()), y=list(base_schedule.values()),
                                 mode='lines+markers', name='Base Schedule',
                                 line=dict(color='#00FF88', width=3), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=list(adjusted_schedule.keys()), y=list(adjusted_schedule.values()),
                                 mode='lines+markers', name=f'Adjusted ({selected_scenario_name})',
                                 line=dict(color='#4682B4', width=3, dash='dot'), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=list(base_schedule.keys()), y=price_curve,
                                 mode='lines+markers', name='Price (€/kWh)',
                                 line=dict(color='#FFA500', width=2, dash='dot'), marker=dict(size=5), yaxis='y2'))

        fig.update_layout(title=f"Scenario Playback: {selected_scenario_name}",
                          template="plotly_dark", height=450, hovermode='x unified', title_x=0.5,
                          xaxis=dict(title="Hour of Day"),
                          yaxis=dict(title="Energy Usage (kWh)"),
                          yaxis2=dict(title="Price (€/kWh)", overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        plot_scenario_adjustments_plotly(solution['x_s'])

    with tab3:
        plot_costs_plotly(scenario_costs_for_display)

    with tab4:
        st.write("**Risk Measures:**")
        var_95 = np.percentile(list(scenario_costs_for_display.values()), 95)
        worst_case = max(scenario_costs_for_display.values())
        st.metric("CVaR α", f"{cvar_alpha:.2f}")
        st.metric("Value at Risk (95%)", f"€{var_95:.2f}")
        st.metric("Worst-case Cost", f"€{worst_case:.2f}")

    with tab5:
        baseline_cost = sum(p_base) * total_demand / 24
        savings = baseline_cost - expected_cost
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost else 0.0
        st.write(f"- Baseline Cost: €{baseline_cost:.2f}")
        st.write(f"- Optimized Cost: €{expected_cost:.2f}")
        st.write(f"- **Savings: €{savings:.2f} ({savings_percent:.1f}%)**")
        if savings > 0:
            st.success("✅ Cost optimization successful!")
        else:
            st.warning("⚠️ No cost savings achieved")

        with st.expander("📄 Detailed Optimization Behavior"):
            st.info(f"""
            **🔍 Realistic Consumer Behavior (Price Uncertainty → Demand Uncertainty):**
            - **Base Coverage Target**: {base_coverage_target:.2f} (fraction of demand)
            - **Price Sensitivity**: {price_sensitivity:.2f} Δdemand for 100% price change
            - **Base Schedule**: {total_usage:.1f} kWh (optimizer's baseline)
            - **Scenario Adjustments**: Applied per scenario based on prices
            - **Low Price Scenario** → uses more energy (cheaper hours)
            - **High Price Scenario** → uses less energy (expensive hours)
            - **Demand Balance**: Base + Adjustments = Scenario demand
            """)
else:
    st.warning("Please upload a valid Excel file.")
