import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PJM Grid Energy Demand Forecast",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------
# LOAD ARTIFACTS (cached so this only runs once)
# --------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_data():
    history = pd.read_csv("processed_history.csv", parse_dates=["date"])
    results = pd.read_csv("test_results.csv", parse_dates=["date"])
    importance = pd.read_csv("feature_importance.csv")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return history, results, importance, metrics


model = load_model()
history, results, importance, metrics = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚡ PJM Grid Energy Demand Forecast")
st.caption(
    "Gradient Boosting model trained on real electricity demand data from "
    "**PJM Interconnection** — a regional grid operator covering parts of "
    "Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, "
    "North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, "
    "and Washington D.C. (Eastern USA), 2002–2018. Forecasts next-day demand "
    "from calendar and lagged-demand features."
)

# --------------------------------------------------
# TOP METRICS ROW
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model RMSE", f"{metrics['gb']['rmse']:.0f} MW")
col2.metric("Model MAE", f"{metrics['gb']['mae']:.0f} MW")
col3.metric("Model R²", f"{metrics['gb']['r2']:.3f}")
col4.metric(
    "Improvement vs. naive baseline",
    f"{metrics['improvement_pct']:.1f}%",
    help="Baseline predicts today's demand = yesterday's demand.",
)

st.divider()

# --------------------------------------------------
# ACTUAL VS PREDICTED CHART (with date range picker)
# --------------------------------------------------

st.subheader("Actual vs Predicted Demand (Test Period)")

min_date, max_date = results["date"].min(), results["date"].max()

date_range = st.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD",
)

mask = (results["date"] >= date_range[0]) & (results["date"] <= date_range[1])
filtered = results.loc[mask]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered["date"], y=filtered["actual_mw"],
    mode="lines", name="Actual", line=dict(color="#333333", width=1.5),
))
fig.add_trace(go.Scatter(
    x=filtered["date"], y=filtered["predicted_mw"],
    mode="lines", name="Predicted (Gradient Boosting)", line=dict(color="#d62728", width=1.5),
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Demand (MW)",
    hovermode="x unified", height=420, margin=dict(t=10),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# SCATTER + FEATURE IMPORTANCE SIDE BY SIDE
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Predicted vs Actual")
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=results["actual_mw"], y=results["predicted_mw"],
        mode="markers", marker=dict(color="#1f77b4", size=5, opacity=0.5),
        name="Test days",
    ))
    lims = [results[["actual_mw", "predicted_mw"]].min().min(),
            results[["actual_mw", "predicted_mw"]].max().max()]
    scatter_fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines", line=dict(color="gray", dash="dash"),
        name="Perfect fit",
    ))
    scatter_fig.update_layout(
        xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)",
        height=380, margin=dict(t=10), showlegend=False,
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

with right:
    st.subheader("Top Feature Importances")
    top15 = importance.head(15).iloc[::-1]
    imp_fig = go.Figure(go.Bar(
        x=top15["Importance"], y=top15["Feature"], orientation="h",
        marker=dict(color="#556b2f"),
    ))
    imp_fig.update_layout(height=380, margin=dict(t=10), xaxis_title="Importance")
    st.plotly_chart(imp_fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# INTERACTIVE "MAKE A PREDICTION" PANEL
# --------------------------------------------------

st.subheader("🔮 Try a Custom Prediction")
st.caption(
    "Pick a day profile and recent demand levels to see what the model forecasts "
    "for the **PJM grid region** (parts of DE, IL, IN, KY, MD, MI, NJ, NC, OH, PA, "
    "TN, VA, WV, and Washington D.C. — Eastern USA). Lag values default to recent "
    "historical averages but can be adjusted."
)

recent = history.tail(30)
default_lag1 = float(recent["demand_mw"].iloc[-1])
default_lag7 = float(recent["demand_mw"].iloc[-7])
default_roll7 = float(recent["demand_mw"].tail(7).mean())

c1, c2, c3 = st.columns(3)
with c1:
    day_of_week = st.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

with c2:
    month = st.slider("Month", 1, 12, 7)
    is_weekend = day_of_week in ("Saturday", "Sunday")
    st.write(f"Weekend: {'Yes' if is_weekend else 'No'}")

with c3:
    demand_lag_1 = st.number_input("Yesterday's demand (MW)", value=round(default_lag1))
    demand_lag_7 = st.number_input("Same day last week (MW)", value=round(default_lag7))
    demand_roll7_mean = st.number_input("7-day rolling average (MW)", value=round(default_roll7))

input_row = pd.DataFrame([{
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "season": season,
    "demand_lag_1": demand_lag_1,
    "demand_lag_7": demand_lag_7,
    "demand_roll7_mean": demand_roll7_mean,
}])

if st.button("Predict demand", type="primary"):
    prediction = model.predict(input_row)[0]
    st.metric("Predicted PJM grid demand", f"{prediction:,.0f} MW")
    st.caption(
        f"For reference: historical PJM demand ranges from roughly "
        f"{history['demand_mw'].min():,.0f} to {history['demand_mw'].max():,.0f} MW "
        f"across the full 2002–2018 dataset."
    )

st.divider()
st.caption(
    "Data: PJM Interconnection hourly grid demand, originally published on Kaggle "
    "(\"Hourly Energy Consumption\" by robikscube). Model trained offline; this app "
    "serves precomputed results and a saved model — it does not retrain live."
)