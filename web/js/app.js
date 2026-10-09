// Page logic. The forecaster runs entirely in the browser: the same module that powers the API.
import * as charts from "./charts.js";
import { Forecaster, dayNumber, isoFromDay } from "./forecast.js";
import { initScenes } from "./scenes.js";

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
  document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="note wrap">Could not load the model data: ${err.message}</p>`);
  throw err;
}
const forecaster = new Forecaster({ model, meta, daily });
const HORIZON = meta.constants.max_horizon;
const day0 = dayNumber(daily.start);

// --- formatting ----------------------------------------------------------------------------------------------
const num = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 0) => (v >= 0 ? "+" : "−") + num(Math.abs(v), d);
const fmtDate = (iso, opts) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { ...opts, timeZone: "UTC" });
const shortDate = (iso) => fmtDate(iso, { weekday: "short", month: "short", day: "numeric" });
const fullDate = (iso) => fmtDate(iso, { weekday: "short", month: "short", day: "numeric", year: "numeric" });

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

// --- scroll-driven scenes + intro ---------------------------------------------------------------------------------------
const scenes = initScenes({ meta, holdout, daily, reduced: REDUCED });

// --- Try it: one timeline, one what-if ------------------------------------------------------------------------------
const startDay = dayNumber(meta.split.test_start);
const maxIndex = forecaster.lastDay - HORIZON - startDay;
const state = { index: dayNumber(meta.constants.default_origin) - startDay, shift: 0 };
const timeInput = $("time");
timeInput.max = maxIndex;
timeInput.value = state.index;

const originOf = () => isoFromDay(startDay + state.index);

function compute() {
  const origin = originOf();
  const opts = { origin, horizon: HORIZON, weather: "actual" };
  const base = forecaster.forecast({ ...opts, tempShift: 0 });
  const scenario = state.shift !== 0 ? forecaster.forecast({ ...opts, tempShift: state.shift }) : null;
  const end = dayNumber(origin) - day0;
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
  const first = (next.scenario ?? next.base)[0];
  $("tag").hidden = !next.scenario;
  $("sub").textContent = `${shortDate(first.date)} · likely between ${num(first.lower_mw)} and ${num(first.upper_mw)} MW`;

  stopTween = tween(animate ? 560 : 0, (t) => {
    let scenario = null;
    if (next.scenario) scenario = t === 1 ? next.scenario : lerpRows(from.scenario ?? from.base, next.scenario, t);
    else if (from.scenario && t < 1) scenario = lerpRows(from.scenario, next.base, t);
    const frame = t === 1
      ? next
      : { context: lerpContext(from.context, next.context, t), base: lerpRows(from.base, next.base, t), scenario };
    current = frame;
    charts.forecastChart($("chart-forecast"), frame.context, frame.base, frame.scenario);
    $("big").textContent = num((frame.scenario ?? frame.base)[0].predicted_mw);
  });
}

function update({ animate = true } = {}) {
  const next = compute();
  $("time-out").textContent = fullDate(originOf());
  const note = $("scenario-note");
  note.hidden = !next.scenario;
  if (next.scenario) {
    const delta = next.scenario.map((r, i) => r.predicted_mw - next.base[i].predicted_mw);
    const avg = delta.reduce((a, b) => a + b, 0) / delta.length;
    const baseMean = next.base.reduce((a, r) => a + r.predicted_mw, 0) / next.base.length;
    note.textContent = `${signed(state.shift, 1)} °C moves the forecast by ${signed(avg)} MW on average (${signed((avg / baseMean) * 100, 1)}%).`;
  }
  paint(next, { animate });
}

let timer = null;
function stopPlay() {
  if (timer === null) return;
  clearInterval(timer);
  timer = null;
  $("play").classList.remove("on");
  $("play").setAttribute("aria-pressed", "false");
  $("play").innerHTML = "&#9654;&ensp;Play";
}
function setIndex(i) {
  state.index = Math.min(Math.max(i, 0), maxIndex);
  timeInput.value = state.index;
  update();
}
function startPlay() {
  if (state.index >= maxIndex) setIndex(0);
  $("play").classList.add("on");
  $("play").setAttribute("aria-pressed", "true");
  $("play").innerHTML = "&#10074;&#10074;&ensp;Pause";
  timer = setInterval(() => (state.index >= maxIndex ? stopPlay() : setIndex(state.index + 1)), 750);
}
$("play").addEventListener("click", () => (timer === null ? startPlay() : stopPlay()));
timeInput.addEventListener("input", () => { stopPlay(); setIndex(Number(timeInput.value)); });

const shiftInput = $("shift");
function setShift(value) {
  state.shift = value;
  shiftInput.value = value;
  $("shift-out").textContent = `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)} °C`;
  document.querySelectorAll(".chips button").forEach((b) => b.classList.toggle("active", Number(b.dataset.shift) === value));
  update();
}
shiftInput.addEventListener("input", (e) => setShift(Number(e.target.value)));
document.querySelectorAll(".chips button").forEach((b) => b.addEventListener("click", () => setShift(Number(b.dataset.shift))));
setShift(0);

// --- limits + footer -----------------------------------------------------------------------------------------------------------
const horizonRow = (day) => meta.horizon.find((r) => r.weather === "climatology" && r.horizon === day);
const limits = [
  `<b>It needs the weather.</b> <span>With only average weather it is ${horizonRow(1).mape.toFixed(0)}% off tomorrow and ${horizonRow(7).mape.toFixed(0)}% off by day 7.</span>`,
  "<b>It stops in August 2018.</b> <span>Newer shifts, like rooftop solar, are not in the data.</span>",
  "<b>It predicts daily averages.</b> <span>Not the peak hour.</span>",
];
$("limit-list").innerHTML = limits.map((html, i) => `<li class="reveal" style="--d:${i * 0.12}s">${html}</li>`).join("");

$("how").innerHTML = `
  <li>Hourly demand for PJM East, averaged to days, plus temperature from five cities.</li>
  <li>${num(meta.model.params.max_iter)} decision trees trained in Python, running here in JavaScript.</li>
  <li>Each day's forecast feeds the next, so a week is forecast one day at a time.</li>
  <li>The range comes from errors on days the model was not trained on.</li>`;
if (meta.links?.repo) $("repo-link").innerHTML = `<a href="${meta.links.repo}">Source code</a>`;
$("try-api").addEventListener("click", async () => {
  const out = $("try-out");
  out.hidden = false;
  out.textContent = "Loading…";
  try {
    const res = await fetch(`/api/forecast?origin=${originOf()}&horizon=3`);
    if (!res.ok || !res.headers.get("content-type")?.includes("json")) throw new Error(`HTTP ${res.status}`);
    out.textContent = JSON.stringify(await res.json(), null, 2);
  } catch (err) {
    out.textContent = `The API is not reachable from this page (${err.message}). It runs as a Cloudflare Worker at /api/*.`;
  }
});

// --- scroll reveals + nav ------------------------------------------------------------------------------------------------------------
const reveal = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add("in");
      reveal.unobserve(e.target);
    }
  }
}, { threshold: 0.2 });
document.querySelectorAll(".reveal").forEach((el, i) => {
  if (!el.style.getPropertyValue("--d")) el.style.setProperty("--d", `${(i % 3) * 0.1}s`);
  reveal.observe(el);
});

const links = [...document.querySelectorAll("[data-nav]")];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.dataset.nav === e.target.id));
  }
}, { rootMargin: "-35% 0px -60% 0px" });
["accuracy", "temperature", "try", "limits"].forEach((id) => spy.observe($(id)));

// Redraw when the system theme changes.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (current) charts.forecastChart($("chart-forecast"), current.context, current.base, current.scenario);
  scenes.redraw();
});
