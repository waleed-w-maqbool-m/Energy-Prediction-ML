// Dashboard logic. The forecaster runs entirely in the browser: the same module that powers the API.
import * as charts from "./charts.js";
import { Forecaster, dayNumber, isoFromDay } from "./forecast.js";

const FINAL = "Hist. gradient boosting + weather (final)";
const NO_WEATHER = "Hist. gradient boosting, no weather";
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

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
const HORIZON = meta.constants.max_horizon;
const day0 = dayNumber(daily.start);

// --- formatting ----------------------------------------------------------------------------------------------
const num = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 0) => (v >= 0 ? "+" : "−") + num(Math.abs(v), d);
const fmtDate = (iso, opts) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { ...opts, timeZone: "UTC" });
const longDate = (iso) => fmtDate(iso, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const shortDate = (iso) => fmtDate(iso, { weekday: "short", day: "numeric", month: "short" });

// --- motion helpers ---------------------------------------------------------------------------------------------
const ease = (t) => 1 - (1 - t) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

/** Run `onFrame(progress)` over `ms`, ending exactly at 1. Reduced motion jumps straight to the end. */
function tween(ms, onFrame) {
  if (REDUCED || ms <= 0) {
    onFrame(1);
    return () => {};
  }
  let raf;
  let start;
  const step = (now) => {
    start ??= now;
    const t = Math.min(1, (now - start) / ms);
    onFrame(t === 1 ? 1 : ease(t));
    if (t < 1) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  return () => cancelAnimationFrame(raf);
}

function onceVisible(el, fn, threshold = 0.3) {
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      fn();
    }
  }, { threshold });
  io.observe(el);
}

// --- hero ribbon: real demand, then the model's forecast, drawn across the page -------------------------------------
function buildRibbon() {
  const W = 1200, H = 120, pad = 6;
  const actual = holdout.actual_mw, predicted = holdout.predicted_mw;
  const all = [...actual, ...predicted];
  const lo = Math.min(...all), hi = Math.max(...all);
  const step = Math.ceil(actual.length / 450);
  const path = (series) => {
    const pts = series.filter((_, i) => i % step === 0);
    return "M" + pts.map((v, i) => `${((i / (pts.length - 1)) * W).toFixed(1)},${(pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad)).toFixed(1)}`).join("L");
  };
  $("ribbon").innerHTML = `<path class="r-actual" pathLength="1" d="${path(actual)}"/><path class="r-model" pathLength="1" d="${path(predicted)}"/>`;
}
buildRibbon();

// --- forecast studio ------------------------------------------------------------------------------------------------------
const state = { origin: meta.constants.default_origin, weather: "actual", shift: 0 };
const originInput = $("origin");
originInput.min = forecaster.firstOriginISO;
originInput.max = forecaster.lastDateISO;
originInput.value = state.origin;

function compute() {
  const opts = { origin: state.origin, horizon: HORIZON, weather: state.weather };
  const base = forecaster.forecast({ ...opts, tempShift: 0 });
  const scenario = state.shift !== 0 ? forecaster.forecast({ ...opts, tempShift: state.shift }) : null;
  const end = dayNumber(state.origin) - day0;
  const dates = [], values = [];
  for (let i = Math.max(0, end - 13); i <= end; i++) {
    dates.push(isoFromDay(day0 + i));
    values.push(daily.demand_mw[i]);
  }
  return { base, scenario, context: { dates, values } };
}

// Interpolation between two forecasts, so changes glide instead of jumping.
const lerpRows = (from, to, t) => {
  if (!from || from.length !== to.length) return to;
  return to.map((r, i) => ({
    ...r,
    predicted_mw: lerp(from[i].predicted_mw, r.predicted_mw, t),
    lower_mw: lerp(from[i].lower_mw, r.lower_mw, t),
    upper_mw: lerp(from[i].upper_mw, r.upper_mw, t),
    temp_mean_c: lerp(from[i].temp_mean_c, r.temp_mean_c, t),
    actual_mw: r.actual_mw !== null && from[i].actual_mw !== null ? lerp(from[i].actual_mw, r.actual_mw, t) : r.actual_mw,
  }));
};
const lerpContext = (from, to, t) =>
  from && from.values.length === to.values.length ? { dates: to.dates, values: to.values.map((v, i) => lerp(from.values[i], v, t)) } : to;

/** Start frame for the first paint: the forecast fans out from the last known value. */
const flatFrame = (next) => {
  const last = next.context.values[next.context.values.length - 1];
  return { context: next.context, scenario: null, base: next.base.map((r) => ({ ...r, predicted_mw: last, lower_mw: last, upper_mw: last })) };
};

let current = null;
let stopTween = () => {};

function paint(next, { animate = true } = {}) {
  stopTween();
  const from = current ?? flatFrame(next);
  const mainNext = (next.scenario ?? next.base)[0];
  const halfWidth = ((mainNext.upper_mw - mainNext.lower_mw) / mainNext.predicted_mw / 2) * 100;
  $("tag").hidden = !next.scenario;
  $("sub").innerHTML = `Next day, ${longDate(mainNext.date)} &middot; 80% range ${num(mainNext.lower_mw)} to ${num(mainNext.upper_mw)} MW (&plusmn;${halfWidth.toFixed(1)}%) &middot; ${mainNext.temp_mean_c.toFixed(1)} &deg;C`;

  stopTween = tween(animate ? 560 : 0, (t) => {
    let scenario = null;
    if (next.scenario) scenario = t === 1 ? next.scenario : lerpRows(from.scenario ?? from.base, next.scenario, t);
    else if (from.scenario && t < 1) scenario = lerpRows(from.scenario, next.base, t);
    const frame = t === 1
      ? next
      : { context: lerpContext(from.context, next.context, t), base: lerpRows(from.base, next.base, t), scenario };
    current = frame;
    charts.forecastChart($("chart-forecast"), frame.context, frame.base, frame.scenario);
    $("big").textContent = `${num((frame.scenario ?? frame.base)[0].predicted_mw)} MW`;
  });
}

function renderTable(frame) {
  const range = (r) => `${num(r.lower_mw)} – ${num(r.upper_mw)}`;
  const head = (cols) => `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
  const body = (rows) => `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
  if (frame.scenario) {
    $("forecast-table").innerHTML = head(["Day", "Temp (&deg;C)", "What-if (MW)", "80% range", "As recorded (MW)", "Change (MW)"]) +
      body(frame.scenario.map((r, i) => [shortDate(r.date), r.temp_mean_c.toFixed(1), num(r.predicted_mw), range(r), num(frame.base[i].predicted_mw), signed(r.predicted_mw - frame.base[i].predicted_mw)]));
  } else {
    $("forecast-table").innerHTML = head(["Day", "Temp (&deg;C)", "Forecast (MW)", "80% range", "Real (MW)", "Error (%)"]) +
      body(frame.base.map((r) => [shortDate(r.date), r.temp_mean_c.toFixed(1), num(r.predicted_mw), range(r),
        r.actual_mw === null ? "–" : num(r.actual_mw),
        r.actual_mw === null ? "–" : signed(((r.predicted_mw - r.actual_mw) / r.actual_mw) * 100, 1)]));
  }
}

function renderNotes(frame) {
  const note = $("scenario-note");
  note.hidden = !frame.scenario;
  if (frame.scenario) {
    const delta = frame.scenario.map((r, i) => r.predicted_mw - frame.base[i].predicted_mw);
    const avg = delta.reduce((a, b) => a + b, 0) / delta.length;
    const baseMean = frame.base.reduce((a, r) => a + r.predicted_mw, 0) / frame.base.length;
    const peak = delta.reduce((best, d, i) => (Math.abs(d) > Math.abs(delta[best]) ? i : best), 0);
    note.innerHTML = `<b>What-if effect:</b> ${signed(state.shift, 1)} &deg;C changes the forecast by ${signed(avg)} MW on average ` +
      `(${signed((avg / baseMean) * 100, 1)}%), up to ${signed(delta[peak])} MW on ${shortDate(frame.scenario[peak].date)}.`;
  }
  const beyond = (frame.scenario ?? frame.base).some((r) => r.weather_source === "climatology") && state.weather === "actual";
  $("data-note").hidden = !beyond;
  if (beyond) $("data-note").textContent = `The data ends on ${longDate(forecaster.lastDateISO)}, so later days use normal seasonal weather and have no real values to compare.`;
}

function update({ animate = true } = {}) {
  let next;
  try {
    next = compute();
  } catch (err) {
    $("sub").textContent = err.message;
    return;
  }
  renderNotes(next);
  renderTable(next);
  state.rows = next.scenario ?? next.base;
  paint(next, { animate });
}

// --- controls -----------------------------------------------------------------------------------------------------------------
let timer = null;
function stopPlay() {
  if (timer === null) return;
  clearInterval(timer);
  timer = null;
  const btn = $("play");
  btn.classList.remove("on");
  btn.setAttribute("aria-pressed", "false");
  btn.innerHTML = "&#9654;&ensp;Play";
}

function setOrigin(iso) {
  const day = Math.min(Math.max(dayNumber(iso), forecaster.firstOrigin), forecaster.lastDay);
  state.origin = isoFromDay(day);
  originInput.value = state.origin;
  update();
}

function startPlay() {
  const lastStart = forecaster.lastDay - HORIZON;
  if (dayNumber(state.origin) >= lastStart) setOrigin(meta.split.test_start);
  const btn = $("play");
  btn.classList.add("on");
  btn.setAttribute("aria-pressed", "true");
  btn.innerHTML = "&#10074;&#10074;&ensp;Pause";
  timer = setInterval(() => {
    const next = dayNumber(state.origin) + 1;
    if (next > lastStart) return stopPlay();
    setOrigin(isoFromDay(next));
  }, 750);
}

$("play").addEventListener("click", () => (timer === null ? startPlay() : stopPlay()));
originInput.addEventListener("change", () => { if (originInput.value) { stopPlay(); setOrigin(originInput.value); } });
$("prev").addEventListener("click", () => { stopPlay(); setOrigin(isoFromDay(dayNumber(state.origin) - 1)); });
$("next").addEventListener("click", () => { stopPlay(); setOrigin(isoFromDay(dayNumber(state.origin) + 1)); });

const shiftInput = $("shift");
function setShift(value) {
  state.shift = value;
  shiftInput.value = value;
  $("shift-out").innerHTML = `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)} &deg;C`;
  document.querySelectorAll(".chips button").forEach((b) => b.classList.toggle("active", Number(b.dataset.shift) === value));
  update();
}
shiftInput.addEventListener("input", (e) => setShift(Number(e.target.value)));
document.querySelectorAll(".chips button").forEach((b) => b.addEventListener("click", () => setShift(Number(b.dataset.shift))));
document.querySelectorAll('input[name="weather"]').forEach((r) => r.addEventListener("change", (e) => {
  state.weather = e.target.value;
  update();
}));
$("download").addEventListener("click", () => {
  const head = "date,predicted_mw,lower_mw,upper_mw,actual_mw,temp_mean_c,weather_source";
  const lines = (state.rows ?? []).map((r) => [r.date, r.predicted_mw, r.lower_mw, r.upper_mw, r.actual_mw ?? "", r.temp_mean_c, r.weather_source].join(","));
  const url = URL.createObjectURL(new Blob([[head, ...lines].join("\n")], { type: "text/csv" }));
  Object.assign(document.createElement("a"), { href: url, download: `pjm_east_forecast_${state.origin}.csv` }).click();
  URL.revokeObjectURL(url);
});
setShift(0);

// --- accuracy -----------------------------------------------------------------------------------------------------------------------------
const byName = Object.fromEntries(meta.comparison.map((r) => [r.model, r]));
const horizonRow = (weather, day) => meta.horizon.find((r) => r.weather === weather && r.horizon === day);
const weatherGain = 100 * (1 - byName[FINAL].rmse / byName[NO_WEATHER].rmse);
const h1Normal = horizonRow("climatology", 1);
const h7Normal = horizonRow("climatology", 7);

$("n-test").textContent = num(meta.split.n_test);
const stats = [
  { to: meta.holdout.mape, dec: 1, label: `typical error on tomorrow's demand. Repeating today's value: ${meta.baseline.mape.toFixed(1)}%.` },
  { to: meta.improvement_pct, dec: 0, label: "lower error than repeating yesterday's demand" },
  { to: weatherGain, dec: 0, label: "of the error removed just by adding temperature" },
];
$("stats").innerHTML = stats.map((s) => `<div class="stat"><div class="n" data-to="${s.to}" data-dec="${s.dec}">0${s.dec ? "." + "0".repeat(s.dec) : ""}%</div><div class="l">${s.label}</div></div>`).join("");
document.querySelectorAll(".stat .n").forEach((el) => {
  const to = Number(el.dataset.to), dec = Number(el.dataset.dec);
  onceVisible(el, () => tween(1300, (t) => { el.textContent = `${(to * t).toFixed(dec)}%`; }), 0.6);
});

$("accuracy-note").textContent =
  `These numbers assume tomorrow's temperature is known, as a weather forecast would provide. With only normal seasonal weather, ` +
  `tomorrow's error is ${h1Normal.mape.toFixed(1)}% (no better than repeating today) and ${h7Normal.mape.toFixed(1)}% by day 7, ` +
  `which still beats repeating last week (${h7Normal.seasonal_naive_mape.toFixed(1)}%).`;

const redraw = [];
function animateChart(id, ms, render, threshold = 0.3) {
  const el = $(id);
  redraw.push(() => render(1));
  onceVisible(el, () => tween(ms, render), threshold);
}
animateChart("chart-compare", 1100, (p) => charts.compareChart($("chart-compare"), meta.comparison, FINAL, p));
animateChart("chart-horizon", 1300, (p) => charts.horizonChart($("chart-horizon"), meta.horizon, p));

// --- temperature -------------------------------------------------------------------------------------------------------------------------------
const allDates = daily.demand_mw.map((_, i) => isoFromDay(day0 + i));
animateChart("chart-scatter", 1400, (p) => charts.scatterChart($("chart-scatter"), allDates, daily.temp_mean, daily.demand_mw, 0.45 * p), 0.25);

// --- about -------------------------------------------------------------------------------------------------------------------------------------------
$("how").innerHTML = `
  <li><b>Data.</b> Hourly demand for PJM East, averaged to days (${num(meta.split.n_train + meta.split.n_test)} days, 2002&ndash;2018), plus temperature from five metro areas.</li>
  <li><b>Model.</b> ${num(meta.model.params.max_iter)} boosted decision trees, trained in Python and evaluated here in JavaScript.</li>
  <li><b>Forecast.</b> Each day's prediction feeds the next, so a week is forecast one day at a time.</li>
  <li><b>Range.</b> The 80% range comes from the model's errors on a two-year window it was not trained on.</li>
  <li><b>Checks.</b> Scored on the last ${num(meta.split.n_test)} days, backtested year by year, and the JavaScript matches the Python model to 1e-9.</li>`;
$("limits").innerHTML = `
  <li><b>Weather is the key input.</b> Without a weather forecast the model is much less accurate (see above).</li>
  <li><b>History ends in August 2018.</b> Newer shifts like rooftop solar or data centres are not captured.</li>
  <li><b>Daily average only.</b> It does not predict the peak hour.</li>`;
if (meta.links?.repo) $("repo-link").innerHTML = `<a href="${meta.links.repo}">Source code on GitHub</a>`;

$("try-api").addEventListener("click", async () => {
  const out = $("try-out");
  out.hidden = false;
  out.textContent = "Loading…";
  try {
    const res = await fetch(`/api/forecast?origin=${state.origin}&horizon=3`);
    if (!res.ok || !res.headers.get("content-type")?.includes("json")) throw new Error(`HTTP ${res.status}`);
    out.textContent = JSON.stringify(await res.json(), null, 2);
  } catch (err) {
    out.textContent = `The API is not reachable from this page (${err.message}). It runs as a Cloudflare Worker at /api/*.`;
  }
});

// --- navigation: highlight the section being read ---------------------------------------------------------------------------------------------------
const links = [...document.querySelectorAll("[data-nav]")];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.dataset.nav === e.target.id));
  }
}, { rootMargin: "-35% 0px -60% 0px" });
document.querySelectorAll(".block").forEach((s) => spy.observe(s));

// Redraw everything when the system theme changes.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (current) charts.forecastChart($("chart-forecast"), current.context, current.base, current.scenario);
  redraw.forEach((fn) => fn());
});
