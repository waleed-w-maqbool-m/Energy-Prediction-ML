// Dashboard logic. The forecaster runs entirely in the browser: the same module that powers the API.
import * as charts from "./charts.js";
import { Forecaster, dayNumber, isoFromDay } from "./forecast.js";

const FINAL = "Hist. gradient boosting + weather (final)";
const NO_WEATHER = "Hist. gradient boosting, no weather";

// --- load ----------------------------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const getJson = (name) => fetch(`data/${name}.json`).then((r) => {
  if (!r.ok) throw new Error(`could not load ${name}.json`);
  return r.json();
});

let model, meta, daily, holdout;
try {
  [model, meta, daily, holdout] = await Promise.all(["model", "meta", "daily", "holdout"].map(getJson));
} catch (err) {
  document.querySelector(".page").insertAdjacentHTML("afterbegin", `<p class="note">Could not load the model data: ${err.message}</p>`);
  throw err;
}
const forecaster = new Forecaster({ model, meta, daily });

// --- formatting -------------------------------------------------------------------------------------------
const num = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 0) => (v >= 0 ? "+" : "−") + num(Math.abs(v), d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const longDate = (iso) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
const shortDate = (iso) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
const table = (el, head, rows, textCols = []) => {
  el.innerHTML = `<thead><tr>${head.map((h, i) => `<th class="${textCols.includes(i) ? "text" : ""}">${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${textCols.includes(i) ? "text" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
};

// --- header ---------------------------------------------------------------------------------------------------------
const byName = Object.fromEntries(meta.comparison.map((r) => [r.model, r]));
const horizonRow = (weather, day) => meta.horizon.find((r) => r.weather === weather && r.horizon === day);
const h1Normal = horizonRow("climatology", 1);
const h7Normal = horizonRow("climatology", 7);
const h7Known = horizonRow("actual", 7);
const weatherGain = 100 * (1 - byName[FINAL].rmse / byName[NO_WEATHER].rmse);

$("scoreboard").innerHTML = `
  <div><div class="n">${meta.holdout.mape.toFixed(1)}%</div>
    <div class="l">typical day-ahead error on ${num(meta.split.n_test)} unseen days (repeating yesterday: ${meta.baseline.mape.toFixed(1)}%)</div></div>
  <div><div class="n">${meta.improvement_pct.toFixed(0)}%</div>
    <div class="l">lower RMSE than the &ldquo;same as yesterday&rdquo; baseline</div></div>
  <div><div class="n">${weatherGain.toFixed(0)}%</div>
    <div class="l">lower RMSE from adding temperature alone</div></div>
  <div><div class="n">${h7Known.mape.toFixed(1)}%</div>
    <div class="l">error on day 7 given a weather forecast (${h7Normal.mape.toFixed(1)}% with only normal seasonal weather)</div></div>`;

// --- forecast studio ---------------------------------------------------------------------------------------------
const state = { origin: meta.constants.default_origin, horizon: meta.constants.max_horizon, weather: "actual", shift: 0 };
const originInput = $("origin");
originInput.min = forecaster.firstOriginISO;
originInput.max = forecaster.lastDateISO;
originInput.value = state.origin;

function renderForecast() {
  let base, scenario = null;
  try {
    const opts = { origin: state.origin, horizon: state.horizon, weather: state.weather };
    base = forecaster.forecast({ ...opts, tempShift: 0 });
    if (state.shift !== 0) scenario = forecaster.forecast({ ...opts, tempShift: state.shift });
  } catch (err) {
    $("result").innerHTML = `<div class="sub">${esc(err.message)}</div>`;
    return;
  }
  const main = scenario ?? base;
  const first = main[0];
  const halfWidth = ((first.upper_mw - first.lower_mw) / first.predicted_mw / 2) * 100;
  $("result").innerHTML =
    `${scenario ? '<span class="tag">Scenario</span>' : ""}<span class="big">${num(first.predicted_mw)} MW</span>` +
    `<div class="sub">Next-day demand forecast: ${longDate(first.date)} &middot; 80% range ${num(first.lower_mw)} to ` +
    `${num(first.upper_mw)} MW (&plusmn;${halfWidth.toFixed(1)}%) &middot; ${first.temp_mean_c.toFixed(1)} &deg;C</div>`;

  // Fourteen days of context behind the forecast origin.
  const end = dayNumber(state.origin) - dayNumber(daily.start);
  const dates = [], values = [];
  for (let i = Math.max(0, end - 13); i <= end; i++) {
    dates.push(isoFromDay(dayNumber(daily.start) + i));
    values.push(daily.demand_mw[i]);
  }
  charts.forecastChart($("chart-forecast"), { dates, values }, base, scenario);

  const note = $("scenario-note");
  note.hidden = !scenario;
  if (scenario) {
    const delta = scenario.map((r, i) => r.predicted_mw - base[i].predicted_mw);
    const avg = delta.reduce((a, b) => a + b, 0) / delta.length;
    const baseMean = base.reduce((a, r) => a + r.predicted_mw, 0) / base.length;
    const peak = delta.reduce((best, d, i) => (Math.abs(d) > Math.abs(delta[best]) ? i : best), 0);
    note.innerHTML = `<b>Scenario effect:</b> ${signed(state.shift, 1)} &deg;C changes the forecast by ${signed(avg)} MW on average ` +
      `(${signed((avg / baseMean) * 100, 1)}%), up to ${signed(delta[peak])} MW on ${shortDate(scenario[peak].date)}.`;
  }
  const dataNote = $("data-note");
  const beyond = main.some((r) => r.weather_source === "climatology") && state.weather === "actual";
  dataNote.hidden = !beyond;
  if (beyond) {
    dataNote.textContent = `The dataset ends on ${longDate(forecaster.lastDateISO)}, so later days use normal seasonal weather and have no actuals to compare.`;
  }

  const range = (r) => `${num(r.lower_mw)} – ${num(r.upper_mw)}`;
  if (scenario) {
    table($("forecast-table"), ["Day", "Temp (&deg;C)", "Forecast (MW)", "80% range (MW)", "Baseline (MW)", "Change (MW)"],
      scenario.map((r, i) => [shortDate(r.date), r.temp_mean_c.toFixed(1), num(r.predicted_mw), range(r), num(base[i].predicted_mw), signed(r.predicted_mw - base[i].predicted_mw)]));
  } else {
    table($("forecast-table"), ["Day", "Temp (&deg;C)", "Forecast (MW)", "80% range (MW)", "Actual (MW)", "Error (%)"],
      base.map((r) => [shortDate(r.date), r.temp_mean_c.toFixed(1), num(r.predicted_mw), range(r),
        r.actual_mw === null ? "–" : num(r.actual_mw),
        r.actual_mw === null ? "–" : signed(((r.predicted_mw - r.actual_mw) / r.actual_mw) * 100, 1)]));
  }
  state.rows = main;
}

let frame = 0;
const scheduleRender = () => {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(renderForecast);
};
originInput.addEventListener("change", () => {
  if (originInput.value) { state.origin = originInput.value; scheduleRender(); }
});
$("horizon").addEventListener("input", (e) => {
  state.horizon = Number(e.target.value);
  $("horizon-out").textContent = state.horizon;
  scheduleRender();
});
$("shift").addEventListener("input", (e) => {
  state.shift = Number(e.target.value);
  $("shift-out").innerHTML = `${state.shift >= 0 ? "+" : "−"}${Math.abs(state.shift).toFixed(1)} &deg;C`;
  scheduleRender();
});
document.querySelectorAll('input[name="weather"]').forEach((r) => r.addEventListener("change", (e) => {
  state.weather = e.target.value;
  scheduleRender();
}));
$("download").addEventListener("click", () => {
  const head = "date,predicted_mw,lower_mw,upper_mw,actual_mw,temp_mean_c,weather_source";
  const lines = (state.rows ?? []).map((r) => [r.date, r.predicted_mw, r.lower_mw, r.upper_mw, r.actual_mw ?? "", r.temp_mean_c, r.weather_source].join(","));
  const url = URL.createObjectURL(new Blob([[head, ...lines].join("\n")], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `pjm_east_forecast_${state.origin}.csv` });
  a.click();
  URL.revokeObjectURL(url);
});

// --- accuracy -------------------------------------------------------------------------------------------------------------
function renderAccuracy() {
  const s = meta.split;
  $("compare-note").textContent = `Every model is fitted on ${s.train_start} to ${s.train_end} and scored on the ${num(s.n_test)} days from ${s.test_start} to ${s.test_end}, which none of them saw during fitting or tuning.`;
  charts.compareChart($("chart-compare"), meta.comparison, FINAL);
  const sorted = [...meta.comparison].sort((a, b) => a.mape - b.mape);
  table($("compare-table"), ["Model", "MAPE (%)", "RMSE (MW)", "MAE (MW)", "R&sup2;", "Notes"],
    sorted.map((r) => [esc(r.model), r.mape.toFixed(2), num(r.rmse), num(r.mae), r.r2.toFixed(3), esc(r.note)]), [0, 5]);
  charts.holdoutChart($("chart-holdout"), holdout);
  charts.horizonChart($("chart-horizon"), meta.horizon);
  const rows = [];
  for (let h = 1; h <= meta.constants.max_horizon; h++) {
    rows.push([h, horizonRow("actual", h).interval_coverage.toFixed(0), horizonRow("climatology", h).interval_coverage.toFixed(0)]);
  }
  table($("coverage-table"), ["Days ahead", "Accurate weather", "Normal weather"], rows);
  charts.yearsChart($("chart-years"), meta.backtest_by_year);
}

// --- drivers ---------------------------------------------------------------------------------------------------------------------
function renderDrivers() {
  charts.importanceChart($("chart-importance"), meta.importance);
  const dates = daily.demand_mw.map((_, i) => isoFromDay(dayNumber(daily.start) + i));
  charts.scatterChart($("chart-scatter"), dates, daily.temp_mean, daily.demand_mw);

  const holidays = new Set(meta.holidays);
  const rows = holdout.date.map((date, i) => {
    const a = holdout.actual_mw[i], p = holdout.predicted_mw[i];
    const dow = (dayNumber(date) + 3) % 7;
    const kind = dow >= 5 ? "Weekend" : holidays.has(date) ? "Weekday holiday" : "Regular weekday";
    return { date, a, p, t: holdout.temp_mean_c[i], err: ((p - a) / a) * 100, kind, month: Number(date.slice(5, 7)) };
  });

  const monthly = Array.from({ length: 12 }, (_, m) => {
    const part = rows.filter((r) => r.month === m + 1);
    return part.reduce((s, r) => s + Math.abs(r.err), 0) / part.length;
  });
  charts.monthsChart($("chart-months"), monthly);

  const kinds = ["Regular weekday", "Weekday holiday", "Weekend"].map((kind) => {
    const part = rows.filter((r) => r.kind === kind);
    return [kind, (part.reduce((s, r) => s + Math.abs(r.err), 0) / part.length).toFixed(2), part.length];
  });
  table($("daytype-table"), ["Day type", "MAPE (%)", "Days"], kinds);

  const worst = [...rows].sort((a, b) => Math.abs(b.err) - Math.abs(a.err)).slice(0, 6);
  table($("worst-table"), ["Day", "Actual (MW)", "Forecast (MW)", "Error (%)", "Temp (&deg;C)"],
    worst.map((r) => [shortDate(r.date) + " " + r.date.slice(0, 4), num(r.a), num(r.p), signed(r.err, 1), r.t.toFixed(1)]));
  const over = worst.filter((r) => r.err > 0).length;
  const temps = worst.map((r) => r.t);
  $("worst-note").textContent = `${over} of these ${worst.length} are over-forecasts, on days between ${Math.round(Math.min(...temps))} and ${Math.round(Math.max(...temps))} °C. That is near the 18.3 °C balance point, where heating and cooling needs are smallest and temperature explains little of the day-to-day variation.`;
}

// --- method ----------------------------------------------------------------------------------------------------------------------------
function renderMethod() {
  const m = meta.model, s = meta.split;
  $("method-text").innerHTML = `
    <h2>How it works</h2>
    <p><b>Data.</b> Hourly demand for the PJM East (PJME) region, averaged to daily load, 2002&ndash;2018
      (${num(s.n_train + s.n_test)} usable days). Temperature is a population-weighted blend of five metro areas
      (Philadelphia, Washington, Baltimore, Newark, Harrisburg) from ERA5 reanalysis.</p>
    <p><b>Features (${m.n_features}).</b> Calendar and U.S. federal holidays; the day's mean, max and min temperature,
      heating and cooling degree days and a 3-day temperature average; demand 1, 2, 7 and 14 days back plus 7- and
      28-day rolling means. Every demand feature uses only days before the one being predicted, which a unit test
      enforces by scrambling future demand.</p>
    <p><b>Model.</b> A gradient-boosted tree ensemble (${num(m.params.max_iter)} trees), tuned by randomized search with
      five-fold time-series cross-validation, trained in Python and exported as JSON. This page evaluates the trees
      in JavaScript, and a test checks the output matches Python to floating-point precision. Multi-day forecasts are
      <i>recursive</i>: each day's prediction becomes the next day's lag input.</p>
    <p><b>Intervals.</b> For each horizon, the 10th and 90th percentiles of relative error on a two-year calibration
      window give an 80% band. Coverage is then measured on the untouched hold-out.</p>
    <p><b>Validation.</b> Chronological hold-out (last 15%), walk-forward yearly backtest, an ablation without
      weather, and comparison to naive, seasonal-naive, linear and the original calendar-only model.</p>
    <h2>Limitations</h2>
    <ul>
      <li><b>Weather is the key input.</b> Headline accuracy assumes the day's temperature is known. With only normal
        seasonal weather, day-ahead error is ${h1Normal.mape.toFixed(1)}% (no better than repeating yesterday) and
        ${h7Normal.mape.toFixed(1)}% by day 7, which still beats repeating last week (${h7Normal.seasonal_naive_mape.toFixed(1)}%).</li>
      <li><b>One region, historical data (to Aug 2018).</b> Newer load shifts (pandemic, data centres, rooftop solar,
        electrification) are not captured, and the app does not pull live data.</li>
      <li><b>Daily average, not peak.</b> Grid operators also care about peak-hour load, which this does not predict.</li>
      <li><b>Five-station temperature index.</b> A load-weighted index over more stations would help further.</li>
    </ul>`;
  $("model-details").innerHTML = `
    <li><b>Trees:</b> ${m.params.max_iter} (depth ${m.params.max_depth}, learning rate ${m.params.learning_rate})</li>
    <li><b>Cross-validated RMSE:</b> ${num(m.cv_rmse)} MW</li>
    <li><b>Trained:</b> ${esc(meta.trained_at.slice(0, 10))} with scikit-learn ${esc(meta.environment["scikit-learn"])}</li>`;
  if (meta.links?.repo) $("repo-link").innerHTML = `<a href="${esc(meta.links.repo)}">Source code on GitHub</a>`;

  $("try-api").addEventListener("click", async () => {
    const out = $("try-out");
    out.hidden = false;
    out.textContent = "Loading…";
    try {
      const res = await fetch(`/api/forecast?origin=${state.origin}&horizon=${Math.min(state.horizon, 3)}`);
      if (!res.ok || !res.headers.get("content-type")?.includes("json")) throw new Error(`HTTP ${res.status}`);
      out.textContent = JSON.stringify(await res.json(), null, 2);
    } catch (err) {
      out.textContent = `The API is not reachable from this page (${err.message}). It is served by the Cloudflare Worker at /api/*.`;
    }
  });
}

// --- tabs ---------------------------------------------------------------------------------------------------------------------------------
const tabs = [...document.querySelectorAll('[role="tab"]')];
const renderers = { accuracy: renderAccuracy, drivers: renderDrivers, method: renderMethod };
const rendered = new Set(["forecast"]);

function showTab(name, { focus = false } = {}) {
  for (const tab of tabs) {
    const active = tab.id === `t-${name}`;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
    $(tab.getAttribute("aria-controls")).hidden = !active;
  }
  const panel = $(`tab-${name}`);
  if (!rendered.has(name)) {
    renderers[name]();
    rendered.add(name);
  } else {
    charts.resizeWithin(panel);
  }
  history.replaceState(null, "", name === "forecast" ? location.pathname : `#${name}`);
}
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => showTab(tab.id.slice(2)));
  tab.addEventListener("keydown", (e) => {
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    showTab(tabs[(next + tabs.length) % tabs.length].id.slice(2), { focus: true });
  });
});

// Redraw everything when the system theme changes.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderForecast();
  for (const name of rendered) if (name !== "forecast") renderers[name]?.();
});

renderForecast();
const initial = location.hash.slice(1);
if (renderers[initial]) showTab(initial);
