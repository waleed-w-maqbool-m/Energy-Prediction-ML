"""Plotly figure builders. All colours come from the active theme palette."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import Palette, rgba

FONT = "IBM Plex Sans, system-ui, sans-serif"


def _base(fig: go.Figure, pal: Palette, height: int, legend: bool = True, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=pal.ink),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color=pal.muted)) if legend else None,
        showlegend=legend, **layout,
    )
    fig.update_xaxes(showgrid=False, linecolor=pal.grid, tickcolor=pal.grid, tickfont=dict(color=pal.muted),
                     title_font=dict(color=pal.muted, size=12))
    fig.update_yaxes(gridcolor=pal.grid, zeroline=False, tickfont=dict(color=pal.muted),
                     title_font=dict(color=pal.muted, size=12))
    return fig


def _band(fig: go.Figure, x, lower, upper, color: str, alpha: float, name: str, rank: int = 1000) -> None:
    fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, alpha), name=name, hoverinfo="skip", legendrank=rank))


# --------------------------------------------------------------------------
# Forecast studio
# --------------------------------------------------------------------------
def forecast_chart(
    context: pd.DataFrame, base: pd.DataFrame, scenario: pd.DataFrame | None, pal: Palette,
    interval_label: str,
) -> go.Figure:
    """Recent actuals, the forecast with its interval, what really happened, and an optional scenario."""
    fig = go.Figure()
    _band(fig, base["date"], base["lower_mw"], base["upper_mw"], pal.model, pal.band_alpha, interval_label, 3)

    actual_x = pd.concat([context["date"], base.loc[base["actual_mw"].notna(), "date"]])
    actual_y = pd.concat([context["demand_mw"], base.loc[base["actual_mw"].notna(), "actual_mw"]])
    fig.add_trace(go.Scatter(x=actual_x, y=actual_y, mode="lines", name="Actual demand",
                             line=dict(color=pal.actual, width=2), hovertemplate="%{y:,.0f} MW", legendrank=1))
    fig.add_trace(go.Scatter(x=base["date"], y=base["predicted_mw"], mode="lines+markers", name="Forecast",
                             line=dict(color=pal.model, width=2.5),
                             marker=dict(size=8, color=pal.model, line=dict(width=2, color=pal.surface)),
                             hovertemplate="%{y:,.0f} MW", legendrank=2))
    if scenario is not None:
        _band(fig, scenario["date"], scenario["lower_mw"], scenario["upper_mw"], pal.scenario,
              pal.band_alpha * 0.8, "Scenario 80% interval", 5)
        fig.add_trace(go.Scatter(x=scenario["date"], y=scenario["predicted_mw"], mode="lines+markers",
                                 name="Scenario forecast", line=dict(color=pal.scenario, width=2.5),
                                 marker=dict(size=8, color=pal.scenario, line=dict(width=2, color=pal.surface)),
                                 hovertemplate="%{y:,.0f} MW", legendrank=4))

    origin = context["date"].iloc[-1]
    fig.add_shape(type="line", x0=origin, x1=origin, yref="paper", y0=0, y1=1,
                  line=dict(color=pal.reference, width=1, dash="dot"))
    fig.add_annotation(x=origin, yref="paper", y=0.0, text="forecast issued", showarrow=False,
                       xanchor="right", xshift=-4, yanchor="bottom", font=dict(size=11, color=pal.muted))
    _base(fig, pal, 400, hovermode="x unified")
    fig.update_layout(legend=dict(orientation="h", y=1.13, x=0, traceorder="normal",
                                  font=dict(size=12, color=pal.muted)))
    fig.update_yaxes(title_text="Daily mean demand (MW)", tickformat=",.0f")
    return fig


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def comparison_chart(comparison: list[dict], final_name: str, pal: Palette) -> go.Figure:
    rows = sorted(comparison, key=lambda r: -r["mape"])  # best model ends up on top
    names = [r["model"] for r in rows]
    colors = [pal.model if n == final_name else pal.reference for n in names]
    fig = go.Figure(go.Bar(
        x=[r["mape"] for r in rows], y=names, orientation="h", marker=dict(color=colors),
        text=[f"{r['mape']:.2f}%" for r in rows], textposition="outside", cliponaxis=False,
        textfont=dict(color=pal.ink, size=13), hovertemplate="%{y}<br>MAPE %{x:.2f}%<extra></extra>",
    ))
    _base(fig, pal, 300, legend=False)
    fig.update_xaxes(title_text="Mean absolute percentage error on the hold-out (lower is better)",
                     range=[0, max(r["mape"] for r in rows) * 1.18])
    fig.update_yaxes(automargin=True, showgrid=False)
    fig.update_layout(bargap=0.38)
    return fig


def holdout_chart(results: pd.DataFrame, pal: Palette) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["date"], y=results["actual_mw"], mode="lines", name="Actual",
                             line=dict(color=pal.actual, width=1.6), hovertemplate="%{y:,.0f} MW"))
    fig.add_trace(go.Scatter(x=results["date"], y=results["predicted_mw"], mode="lines", name="Model, day-ahead",
                             line=dict(color=pal.model, width=1.6), hovertemplate="%{y:,.0f} MW"))
    _base(fig, pal, 340, hovermode="x unified")
    fig.update_yaxes(title_text="Daily mean demand (MW)", tickformat=",.0f")
    return fig


def horizon_chart(horizon: pd.DataFrame, pal: Palette) -> go.Figure:
    fig = go.Figure()
    known = horizon[horizon["weather"] == "actual"].sort_values("horizon")
    unknown = horizon[horizon["weather"] == "climatology"].sort_values("horizon")
    seasonal = known
    fig.add_trace(go.Scatter(x=seasonal["horizon"], y=seasonal["seasonal_naive_mape"], mode="lines+markers",
                             name="Last week's value", line=dict(color=pal.reference, width=2),
                             marker=dict(size=8), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=unknown["horizon"], y=unknown["mape"], mode="lines+markers",
                             name="Model, normal weather assumed", line=dict(color=pal.scenario, width=2),
                             marker=dict(size=8), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=known["horizon"], y=known["mape"], mode="lines+markers",
                             name="Model, weather known", line=dict(color=pal.model, width=2.5),
                             marker=dict(size=8), hovertemplate="%{y:.2f}%"))
    _base(fig, pal, 340, hovermode="x unified")
    fig.update_layout(margin=dict(t=56), legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12, color=pal.muted)))
    fig.update_xaxes(title_text="Days ahead", dtick=1)
    fig.update_yaxes(title_text="MAPE (%)", rangemode="tozero")
    return fig


def yearly_chart(by_year: pd.DataFrame, pal: Palette) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_year["year"], y=by_year["naive_mape"], name="Same as yesterday",
                         marker=dict(color=pal.reference), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Bar(x=by_year["year"], y=by_year["model_mape"], name="Model",
                         marker=dict(color=pal.model), hovertemplate="%{y:.2f}%"))
    _base(fig, pal, 320, barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="x unified")
    fig.update_xaxes(dtick=1, title_text="Test year (model retrained on all earlier years)")
    fig.update_yaxes(title_text="MAPE (%)", rangemode="tozero")
    return fig


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def importance_chart(importance: pd.DataFrame, pal: Palette, top: int = 12) -> go.Figure:
    top_df = importance.head(top).iloc[::-1]
    group_color = {"Demand history": pal.model, "Weather": pal.scenario, "Calendar": pal.reference}
    fig = go.Figure()
    for group, color in group_color.items():
        part = top_df[top_df["group"] == group]
        fig.add_trace(go.Bar(
            x=part["rmse_increase_mw"], y=part["feature"], orientation="h", name=group,
            marker=dict(color=color), error_x=dict(type="data", array=part["std"], color=pal.muted, thickness=1),
            hovertemplate="%{y}<br>+%{x:,.0f} MW RMSE when shuffled<extra></extra>",
        ))
    _base(fig, pal, 420, barmode="stack", bargap=0.3)
    fig.update_xaxes(title_text="Increase in RMSE (MW) when the feature is shuffled")
    fig.update_yaxes(showgrid=False, automargin=True, categoryorder="array",
                     categoryarray=list(top_df["feature"]))
    return fig


def temperature_scatter(daily: pd.DataFrame, pal: Palette) -> go.Figure:
    """Demand vs. temperature: the V-shape that justifies degree-day features."""
    nonworking = daily["date"].dt.dayofweek >= 5
    fig = go.Figure()
    for mask, name, color in ((~nonworking, "Weekdays", pal.model), (nonworking, "Weekends", pal.scenario)):
        part = daily[mask]
        fig.add_trace(go.Scattergl(
            x=part["temp_mean"], y=part["demand_mw"], mode="markers", name=name,
            marker=dict(size=5, color=rgba(color, 0.45)), hovertemplate="%{x:.1f} °C: %{y:,.0f} MW",
        ))
    _base(fig, pal, 380)
    fig.update_layout(legend=dict(itemsizing="constant", orientation="h", y=1.1, x=0,
                                  font=dict(size=12, color=pal.muted)))
    fig.update_xaxes(title_text="Daily mean temperature, region average (°C)")
    fig.update_yaxes(title_text="Daily mean demand (MW)", tickformat=",.0f")
    return fig


def month_error_chart(results: pd.DataFrame, pal: Palette) -> go.Figure:
    err = (results["actual_mw"] - results["predicted_mw"]).abs() / results["actual_mw"] * 100
    by_month = err.groupby(results["date"].dt.month).mean().reindex(range(1, 13))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Bar(x=months, y=by_month.values, marker=dict(color=pal.model),
                           hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    _base(fig, pal, 300, legend=False, bargap=0.35)
    fig.update_yaxes(title_text="MAPE (%)", rangemode="tozero")
    return fig


def daytype_table(results: pd.DataFrame) -> pd.DataFrame:
    from energy_forecast.features import nonworking

    pct = (results["actual_mw"] - results["predicted_mw"]).abs() / results["actual_mw"] * 100
    kind = np.where(results["date"].dt.dayofweek >= 5, "Weekend",
                    np.where(nonworking(results["date"]), "Weekday holiday", "Regular weekday"))
    out = pct.groupby(kind).agg(["mean", "count"]).rename(columns={"mean": "MAPE (%)", "count": "Days"})
    return out.round({"MAPE (%)": 2}).reset_index(names="Day type")
