"""PJM East demand forecast: Streamlit dashboard.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, theme
from energy_forecast import config
from energy_forecast.forecast import Forecaster

st.set_page_config(page_title="PJM East Demand Forecast", page_icon="⚡", layout="wide")
theme.inject_css()
pal = theme.palette()

FINAL = "Hist. gradient boosting + weather (final)"
NO_WEATHER = "Hist. gradient boosting, no weather"


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_forecaster() -> Forecaster:
    return Forecaster()


@st.cache_data
def load_tables():
    art = config.ARTIFACT_DIR
    return (
        pd.read_csv(art / "test_results.csv", parse_dates=["date"]),
        pd.read_csv(art / "feature_importance.csv"),
        pd.read_csv(art / "backtest_by_year.csv"),
        pd.read_csv(art / "horizon_metrics.csv"),
    )


@st.cache_data(show_spinner=False)
def run_forecast(origin: str, horizon: int, weather: str, shift: float) -> pd.DataFrame:
    return load_forecaster().forecast(origin, horizon, weather, shift)


fc = load_forecaster()
meta = fc.meta
results, importance, by_year, horizon_df = load_tables()
by_name = {r["model"]: r for r in meta["comparison"]}
hold, naive = meta["holdout"], meta["baseline"]
split = meta["split"]


def horizon_row(weather: str, day: int) -> pd.Series:
    return horizon_df[(horizon_df["weather"] == weather) & (horizon_df["horizon"] == day)].iloc[0]


h1_normal, h7_normal = horizon_row("climatology", 1), horizon_row("climatology", 7)
h7_known = horizon_row("actual", 7)
weather_gain = 100 * (1 - by_name[FINAL]["rmse"] / by_name[NO_WEATHER]["rmse"])


# --------------------------------------------------------------------------
# Masthead + scoreboard
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
      <h1>PJM East electricity demand, forecast a week ahead</h1>
      <p>A gradient-boosting model that combines temperature, calendar and recent demand to predict
      daily grid load, with prediction intervals calibrated on held-out data.
      Trained on 2002–2018 data for the eastern part of the largest U.S. grid operator.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="scoreboard">
      <div><div class="n">{hold['mape']:.1f}%</div>
        <div class="l">typical day-ahead error on {split['n_test']:,} unseen days
        (repeating yesterday: {naive['mape']:.1f}%)</div></div>
      <div><div class="n">{meta['improvement_pct']:.0f}%</div>
        <div class="l">lower RMSE than the “same as yesterday” baseline</div></div>
      <div><div class="n">{weather_gain:.0f}%</div>
        <div class="l">lower RMSE from adding temperature alone</div></div>
      <div><div class="n">{h7_known['mape']:.1f}%</div>
        <div class="l">error on day 7 given a weather forecast ({h7_normal['mape']:.1f}% with only normal
        seasonal weather)</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_forecast, tab_accuracy, tab_drivers, tab_method = st.tabs(
    ["Forecast studio", "Accuracy", "What drives demand", "Method and API"]
)

# --------------------------------------------------------------------------
# Forecast studio
# --------------------------------------------------------------------------
with tab_forecast:
    peak_day = results.loc[results["actual_mw"].idxmax(), "date"]
    default_origin = (peak_day - pd.Timedelta(days=3)).date()

    controls, view = st.columns([1, 3], gap="large")
    with controls:
        origin = st.date_input(
            "Last known day",
            value=default_origin,
            min_value=fc.first_origin.date(),
            max_value=fc.last_date.date(),
            help="The forecast is issued at the end of this day, using demand up to and including it. "
            "The default is three days before the busiest day in the hold-out period.",
        )
        horizon = st.slider("Days ahead", 1, config.MAX_HORIZON, 7)
        weather_label = st.radio(
            "Weather assumption",
            ["Accurate weather forecast", "Normal weather for the season"],
            help="The model needs each day's temperature. In production it would come from a weather "
            "forecast. 'Accurate' uses the temperatures that actually occurred; 'normal' uses the "
            "seasonal average, which is what you would have with no weather forecast at all "
            "(wider intervals, larger errors).",
        )
        weather = "actual" if weather_label.startswith("Accurate") else "climatology"
        shift = st.slider(
            "Temperature scenario (°C)", -8.0, 8.0, 0.0, 0.5, format="%+.1f",
            help="Adds this many degrees to every forecast day, and re-runs the full recursive "
            "forecast. Try +5 °C in July for a heatwave or −8 °C in January for a cold snap.",
        )

    with view:
        base = run_forecast(str(origin), horizon, weather, 0.0)
        scenario = run_forecast(str(origin), horizon, weather, float(shift)) if shift else None
        main = scenario if scenario is not None else base

        first = main.iloc[0]
        tag = '<span class="tag scenario">Scenario</span>' if scenario is not None else ""
        pct_lo = (first["upper_mw"] - first["lower_mw"]) / first["predicted_mw"] / 2 * 100
        sub = (
            f"{first['date']:%A %d %B %Y}  ·  80% range {first['lower_mw']:,.0f} to {first['upper_mw']:,.0f} MW "
            f"(±{pct_lo:.1f}%)  ·  {first['temp_mean_c']:.1f} °C"
        )
        st.markdown(
            f'<div class="result">{tag}<span class="big">{first["predicted_mw"]:,.0f} MW</span>'
            f'<div class="sub">Next-day demand forecast: {sub}</div></div>',
            unsafe_allow_html=True,
        )

        origin_ts = pd.Timestamp(origin)
        context = fc.daily[
            (fc.daily["date"] <= origin_ts) & (fc.daily["date"] > origin_ts - pd.Timedelta(days=14))
        ]
        st.plotly_chart(charts.forecast_chart(context, base, scenario, pal, "80% interval"),
                        width="stretch", config={"displayModeBar": False})

        if scenario is not None:
            delta = scenario["predicted_mw"] - base["predicted_mw"]
            peak = delta.abs().idxmax()
            st.markdown(
                f'<div class="note"><b>Scenario effect:</b> {shift:+.1f} °C changes the forecast by '
                f"{delta.mean():+,.0f} MW on average ({delta.mean() / base['predicted_mw'].mean() * 100:+.1f}%), "
                f"up to {delta[peak]:+,.0f} MW on {scenario.loc[peak, 'date']:%a %d %b}.</div>",
                unsafe_allow_html=True,
            )
        if (main["weather_source"] == "climatology").any() and weather == "actual":
            st.info(
                f"The dataset ends on {fc.last_date:%d %b %Y}, so later days use normal seasonal weather "
                "and have no actuals to compare."
            )

        pretty = {"temp_mean_c": "Temp (°C)", "predicted_mw": "Forecast (MW)"}
        shared = {
            "Temp (°C)": st.column_config.NumberColumn(format="%.1f"),
            "Forecast (MW)": st.column_config.NumberColumn(format="%,.0f"),
            "Range": st.column_config.TextColumn("80% range (MW)"),
        }
        table = main.assign(
            Day=main["date"].dt.strftime("%a %d %b"),
            Range=main.apply(lambda r: f"{r['lower_mw']:,.0f} – {r['upper_mw']:,.0f}", axis=1),
        ).rename(columns=pretty)
        if scenario is not None:
            # What-if runs have no "truth" to compare with; show the change against the unshifted forecast.
            table["Baseline (MW)"] = base["predicted_mw"].to_numpy()
            table["Change (MW)"] = table["Forecast (MW)"] - table["Baseline (MW)"]
            columns = ["Day", "Temp (°C)", "Forecast (MW)", "Range", "Baseline (MW)", "Change (MW)"]
            config_cols = {
                **shared,
                "Baseline (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Change (MW)": st.column_config.NumberColumn(format="%+,.0f"),
            }
        else:
            table["Actual (MW)"] = main["actual_mw"]
            table["Error"] = (main["predicted_mw"] - main["actual_mw"]) / main["actual_mw"] * 100
            columns = ["Day", "Temp (°C)", "Forecast (MW)", "Range", "Actual (MW)", "Error"]
            config_cols = {
                **shared,
                "Actual (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Error": st.column_config.NumberColumn("Error (%)", format="%+.1f"),
            }
        st.dataframe(table[columns], hide_index=True, width="stretch", column_config=config_cols)
        st.download_button(
            "Download forecast (CSV)", main.to_csv(index=False).encode(),
            file_name=f"pjm_east_forecast_{origin}.csv", mime="text/csv",
        )

# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
with tab_accuracy:
    st.markdown("### How it compares")
    st.caption(
        f"Every model is fitted on {split['train_start']} to {split['train_end']} and scored on the "
        f"{split['n_test']:,} days from {split['test_start']} to {split['test_end']}, which none of them saw "
        "during fitting or tuning."
    )
    st.plotly_chart(charts.comparison_chart(meta["comparison"], FINAL, pal), width="stretch",
                    config={"displayModeBar": False})
    cmp_df = pd.DataFrame(meta["comparison"]).sort_values("mape").rename(columns={
        "model": "Model", "mape": "MAPE (%)", "rmse": "RMSE (MW)", "mae": "MAE (MW)", "r2": "R²", "note": "Notes",
    })
    st.dataframe(cmp_df[["Model", "MAPE (%)", "RMSE (MW)", "MAE (MW)", "R²", "Notes"]], hide_index=True,
                 width="stretch", column_config={
                     "MAPE (%)": st.column_config.NumberColumn(format="%.2f"),
                     "RMSE (MW)": st.column_config.NumberColumn(format="%,.0f"),
                     "MAE (MW)": st.column_config.NumberColumn(format="%,.0f"),
                     "R²": st.column_config.NumberColumn(format="%.3f"),
                 })

    st.markdown("### Day-ahead forecasts against actual demand")
    lo, hi = results["date"].min().to_pydatetime(), results["date"].max().to_pydatetime()
    window = st.slider("Date range", lo, hi, (lo, hi), format="YYYY-MM-DD", label_visibility="collapsed")
    shown = results[(results["date"] >= window[0]) & (results["date"] <= window[1])]
    st.plotly_chart(charts.holdout_chart(shown, pal), width="stretch", config={"displayModeBar": False})

    st.markdown("### Accuracy by forecast horizon")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(charts.horizon_chart(horizon_df, pal), width="stretch", config={"displayModeBar": False})
    with right:
        cover = horizon_df.pivot(index="horizon", columns="weather", values="interval_coverage").round(0)
        cover.columns = [{"actual": "Accurate weather", "climatology": "Normal weather"}[c] for c in cover.columns]
        st.markdown("**How often the 80% interval contains the actual value (%)**")
        st.dataframe(cover.rename_axis("Days ahead"), width="stretch")
        st.markdown(
            '<div class="note">Intervals are calibrated on a two-year window before the hold-out and then '
            "checked on the hold-out. They are somewhat too narrow when the weather is assumed rather than "
            "known, which is why that case is reported separately.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Stability across years")
    st.plotly_chart(charts.yearly_chart(by_year, pal), width="stretch", config={"displayModeBar": False})
    st.caption("Walk-forward backtest: for each year the model is retrained on everything before it, "
               "then scored one day ahead on that year.")

# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
with tab_drivers:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Which inputs matter")
        st.plotly_chart(charts.importance_chart(importance, pal), width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="note">Permutation importance on the hold-out: how much worse the forecast gets when one '
            "input is randomly shuffled. Recent demand sets the level; temperature explains the day-to-day "
            "swings that yesterday's demand cannot.</div>", unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Demand against temperature")
        st.plotly_chart(charts.temperature_scatter(fc.daily, pal), width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="note">Demand rises in both directions away from a mild day: heating in winter, '
            "air conditioning in summer. Heating and cooling degree days (distance from 18.3 °C) give the model "
            "this V-shape directly.</div>", unsafe_allow_html=True,
        )

    st.markdown("### Where the model is weakest")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(charts.month_error_chart(results, pal), width="stretch", config={"displayModeBar": False})
        st.caption("Mean absolute percentage error by calendar month, hold-out period.")
    with c2:
        st.dataframe(charts.daytype_table(results), hide_index=True, width="stretch")
        scored = results.assign(
            error_pct=(results["predicted_mw"] - results["actual_mw"]) / results["actual_mw"] * 100,
            day=results["date"].dt.strftime("%a %d %b %Y"),
        )
        worst = scored.loc[scored["error_pct"].abs().sort_values(ascending=False).index].head(6)
        st.markdown("**Largest misses**")
        st.dataframe(
            worst[["day", "actual_mw", "predicted_mw", "error_pct", "temp_mean_c"]].rename(columns={
                "day": "Day", "actual_mw": "Actual (MW)", "predicted_mw": "Forecast (MW)",
                "error_pct": "Error (%)", "temp_mean_c": "Temp (°C)"}),
            hide_index=True, width="stretch",
            column_config={
                "Actual (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Forecast (MW)": st.column_config.NumberColumn(format="%,.0f"),
                "Error (%)": st.column_config.NumberColumn(format="%+.1f"),
                "Temp (°C)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        over = int((worst["error_pct"] > 0).sum())
        lo_t, hi_t = worst["temp_mean_c"].min(), worst["temp_mean_c"].max()
        st.markdown(
            f'<div class="note">{over} of these {len(worst)} are over-forecasts, on days between {lo_t:.0f} and '
            f"{hi_t:.0f} °C. That is near the 18.3 °C balance point, where heating and cooling needs are smallest "
            "and temperature explains little of the day-to-day variation.</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Method
# --------------------------------------------------------------------------
with tab_method:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            f"""
### How it works

**Data.** Hourly demand for the PJM East (PJME) region, averaged to daily load, 2002–2018
({split['n_train'] + split['n_test']:,} usable days). Temperature is a population-weighted blend of five
metro areas (Philadelphia, Washington, Baltimore, Newark, Harrisburg) from ERA5 reanalysis.

**Features ({meta['model']['n_features']}).** Calendar and U.S. federal holidays; today's mean, max and min
temperature, heating and cooling degree days and a 3-day temperature average; demand 1, 2, 7 and 14 days
back plus 7- and 28-day rolling means. Every demand feature uses only days before the one being predicted,
which a unit test enforces by scrambling future demand.

**Model.** `HistGradientBoostingRegressor`, tuned by randomized search with five-fold time-series
cross-validation. Multi-day forecasts are *recursive*: each day's prediction becomes the next day's lag input.

**Intervals.** For each horizon, the 10th and 90th percentiles of relative error on a two-year calibration
window give an 80% band. Coverage is then measured on the untouched hold-out.

**Validation.** Chronological hold-out (last 15%), walk-forward yearly backtest, ablation without weather,
and comparison to naive, seasonal-naive, linear and the original calendar-only model.

### Limitations

- **Weather is the key input.** Headline accuracy assumes the day's temperature is known. With only normal
  seasonal weather, day-ahead error is {h1_normal['mape']:.1f}% (no better than repeating yesterday) and
  {h7_normal['mape']:.1f}% by day 7, which still beats repeating last week ({h7_normal['seasonal_naive_mape']:.1f}%).
- **One region, historical data (to Aug 2018).** Newer load shifts (pandemic, data centres, rooftop solar,
  electrification) are not captured, and the app does not pull live data.
- **Daily average, not peak.** Grid operators also care about peak-hour load, which this does not predict.
- **Five-station temperature index.** A proper load-weighted index over more stations would help further.
"""
        )
    with right:
        st.markdown("### REST API")
        st.markdown("The same forecaster is served by a FastAPI service:")
        st.code(
            "uvicorn api.main:app\n"
            "\n"
            'curl "localhost:8000/forecast?origin=2017-07-10"\n'
            "\n"
            "curl localhost:8000/model",
            language="bash", wrap_lines=True,
        )
        st.caption("`POST /predict` takes your own recent demand history and a temperature.")
        st.markdown("Interactive docs are at `/docs`.")
        st.markdown("### Model details")
        params = meta["model"]["params"]
        st.markdown(
            f"- **Trees:** {params['max_iter']} (depth {params['max_depth']}, lr {params['learning_rate']})\n"
            f"- **Cross-validated RMSE:** {meta['model']['cv_rmse']:,.0f} MW\n"
            f"- **Trained:** {meta['trained_at'][:10]} with scikit-learn {meta['environment']['scikit-learn']}"
        )
        if config.REPO_URL:
            st.markdown(f"[Source code on GitHub]({config.REPO_URL})")

st.markdown(
    '<div class="note" style="margin-top:2.5rem">Demand data: PJM Interconnection via the Kaggle dataset '
    '“Hourly Energy Consumption” (robikscube). Weather data by '
    '<a href="https://open-meteo.com/">Open-Meteo.com</a> (ERA5, CC BY 4.0).</div>',
    unsafe_allow_html=True,
)
