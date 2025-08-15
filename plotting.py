import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ------------------------
# EXPECTED COST CALCULATION
# ------------------------
def compute_expected_cost(scenario_costs, probabilities):
    return sum(probabilities[s] * scenario_costs[s] for s in scenario_costs)

# ------------------------
# COSTS BAR CHART
# ------------------------
def plot_costs_plotly(scenario_costs):
    scenarios = [f"Scenario {s+1}" for s in scenario_costs.keys()]
    costs = list(scenario_costs.values())
    colors = ['#00FF88' if c == min(costs) else '#FF6B6B' if c == max(costs) else '#4682B4' for c in costs]

    fig = go.Figure(
        data=[go.Bar(
            x=scenarios,
            y=costs,
            text=[f"€{c:.2f}" for c in costs],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Cost: €%{y:.2f}<extra></extra>'
        )]
    )
    avg_cost = np.mean(costs)
    fig.add_hline(y=avg_cost, line_dash="dash", line_color="yellow", annotation_text=f"Avg: €{avg_cost:.2f}")
    fig.update_layout(
        title="Scenario-wise Cost Comparison",
        xaxis_title="Scenario",
        yaxis_title="Total Cost (€)",
        template="plotly_dark",
        height=400,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# SCENARIO ADJUSTMENTS HEATMAP
# ------------------------
def plot_scenario_adjustments_plotly(x_s_dict):
    df = pd.DataFrame(x_s_dict).T
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=[f"H{h+1}" for h in df.columns],
            y=[f"Scenario {int(s)+1}" for s in df.index],
            colorscale='RdBu_r',
            colorbar=dict(title='kWh'),
            zmid=0,
            hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Adj: %{z:.2f} kWh<extra></extra>'
        )
    )
    fig.update_layout(
        title="Scenario Adjustments Heatmap",
        xaxis_title="Hour",
        yaxis_title="Scenario",
        template="plotly_dark",
        height=350,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# BASE SCHEDULE WITH PRICE OVERLAY
# ------------------------
def plot_base_schedule_plotly(x_dict, p_base=None):
    hours = list(x_dict.keys())
    values = list(x_dict.values())

    fig = go.Figure()

    # Energy usage line
    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode='lines+markers',
        name='Energy Usage',
        line=dict(color='#00FF88', width=3, shape='spline'),
        marker=dict(size=8, color='#00FF88', line=dict(width=2, color='#00CC6A')),
        hovertemplate='Hour %{x}: %{y:.2f} kWh'
    ))

    # Area fill for usage
    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.3)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Price overlay (secondary y-axis)
    if p_base:
        fig.add_trace(go.Scatter(
            x=hours,
            y=p_base,
            mode='lines+markers',
            name='Price (€/kWh)',
            line=dict(color='#FFA500', width=2, dash='dot'),
            marker=dict(size=6, color='#FFA500'),
            hovertemplate='Hour %{x}: %{y:.4f} €/kWh',
            yaxis='y2'
        ))

    # Layout
    fig.update_layout(
        title="Optimized Energy Schedule with Price Overlay",
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        title_x=0.5,
        xaxis=dict(title="Hour of Day", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Energy Usage (kWh)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Price (€/kWh)", overlaying='y', side='right', showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)
