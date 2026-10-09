// Scroll-driven scenes. Each scene is a tall section with a sticky stage; the scroll position
// inside the section (0..1) drives the animation, so scrolling is the playback control.
import { rgba, theme } from "./charts.js";

const $ = (id) => document.getElementById(id);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const easeOut = (t) => 1 - (1 - t) ** 3;
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2);
const BAR_MAX = 6.5; // axis maximum (%) shared by both bars

/** SVG path for a series scaled into a W x H box. */
function linePath(series, W, H, lo, hi, pad = 6, maxPoints = 450) {
  const step = Math.max(1, Math.ceil(series.length / maxPoints));
  const pts = series.filter((_, i) => i % step === 0);
  const d = pts.map((v, i) => `${((i / (pts.length - 1)) * W).toFixed(1)},${(pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad)).toFixed(1)}`);
  return `M${d.join("L")}`;
}

/**
 * Line chart in real pixel coordinates (viewBox = element size, rebuilt on resize). Pixel space keeps
 * dash-based "draw on" animation exact; a stretched viewBox would distort dash lengths.
 */
function lineChart(svg, specs, lo, hi, pad) {
  svg.innerHTML = specs.map((s) => `<path class="${s.cls}" pathLength="1"/>`).join("");
  const paths = [...svg.querySelectorAll("path")];
  const layout = () => {
    const W = svg.clientWidth, H = svg.clientHeight;
    if (!W || !H) return;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    paths.forEach((path, i) => path.setAttribute("d", linePath(specs[i].series, W, H, lo, hi, pad)));
  };
  layout();
  new ResizeObserver(layout).observe(svg);
  return paths;
}

// --- intro line ------------------------------------------------------------------------------------------
function buildHeroLine(holdout) {
  const a = holdout.actual_mw;
  lineChart($("hero-line"), [{ cls: "", series: a }], Math.min(...a), Math.max(...a), 6);
}

// --- scene 1: the model's line draws over real demand as you scroll --------------------------------------
function accuracyScene(meta, holdout) {
  const a = holdout.actual_mw, p = holdout.predicted_mw;
  const all = [...a, ...p];
  const lo = Math.min(...all), hi = Math.max(...all);
  const [actualPath, modelPath] = lineChart($("ribbon"), [
    { cls: "r-actual", series: a },
    { cls: "r-model", series: p },
  ], lo, hi, 8);
  $("n-days").textContent = meta.split.n_test.toLocaleString("en-US");
  const model = meta.holdout.mape, base = meta.baseline.mape;

  return (t) => {
    actualPath.style.strokeDashoffset = 1 - easeOut(clamp01(t / 0.18));
    const drawn = easeInOut(clamp01((t - 0.12) / 0.6));
    modelPath.style.strokeDashoffset = 1 - drawn;
    const n = model * drawn;
    $("score-n").textContent = n.toFixed(1);
    $("bar-model").style.width = `${(n / BAR_MAX) * 100}%`;
    $("bar-model-n").textContent = `${n.toFixed(1)}%`;
    const b = base * easeOut(clamp01((t - 0.5) / 0.28));
    $("bar-base").style.width = `${(b / BAR_MAX) * 100}%`;
    $("bar-base-n").textContent = b > 0.05 ? `${b.toFixed(1)}%` : "";
  };
}

// --- scene 2: demand against temperature, swept from cold to hot -----------------------------------------
function temperatureScene(daily) {
  const canvas = $("scatter");
  const order = daily.temp_mean.map((_, i) => i).sort((i, j) => daily.temp_mean[i] - daily.temp_mean[j]);
  const xs = order.map((i) => daily.temp_mean[i]);
  const ys = order.map((i) => daily.demand_mw[i]);
  const startDay = new Date(`${daily.start}T00:00:00Z`).getTime() / 86_400_000;
  const weekend = order.map((i) => (startDay + i + 3) % 7 >= 5); // Monday = 0
  const N = xs.length;
  const xmin = Math.floor(xs[0] / 5) * 5, xmax = Math.ceil(xs[N - 1] / 5) * 5;
  const ymin = Math.min(...ys) - 1500, ymax = Math.max(...ys) + 1500;
  let last = 0;

  const render = (t) => {
    last = t;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const c = theme();
    const L = 52, R = 12, T = 28, B = 30;
    const X = (v) => L + ((v - xmin) / (xmax - xmin)) * (w - L - R);
    const Y = (v) => h - B - ((v - ymin) / (ymax - ymin)) * (h - T - B);

    ctx.font = `12px ${c.font}`;
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (let v = Math.ceil(ymin / 10000) * 10000; v <= ymax; v += 10000) {
      ctx.strokeStyle = c.grid;
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke();
      ctx.fillStyle = c.muted; ctx.textAlign = "right";
      ctx.fillText(`${v / 1000}k`, L - 8, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let v = Math.ceil(xmin / 10) * 10; v <= xmax; v += 10) {
      ctx.fillStyle = c.muted;
      ctx.fillText(`${v}°C`, X(v), h - B + 8);
    }

    const count = Math.floor(easeInOut(clamp01((t - 0.05) / 0.82)) * N);
    for (const flag of [false, true]) {
      ctx.fillStyle = rgba(flag ? c.scenario : c.model, 0.5);
      for (let k = 0; k < count; k++) {
        if (weekend[k] !== flag) continue;
        ctx.beginPath(); ctx.arc(X(xs[k]), Y(ys[k]), 2.1, 0, 6.2832); ctx.fill();
      }
    }
    if (count > 0 && count < N) {
      const fx = X(xs[count - 1]);
      ctx.strokeStyle = c.reference; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(fx, T - 6); ctx.lineTo(fx, h - B); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.ink; ctx.textAlign = fx > w - 70 ? "right" : "left"; ctx.textBaseline = "bottom";
      ctx.fillText(`${xs[count - 1].toFixed(0)}°C`, fx + (fx > w - 70 ? -6 : 6), T);
    }
    // legend
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    [["Weekdays", c.model], ["Weekends", c.scenario]].forEach(([label, color], i) => {
      const x = L + i * 96;
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, 10, 4, 0, 6.2832); ctx.fill();
      ctx.fillStyle = c.muted; ctx.fillText(label, x + 10, 10);
    });
  };
  new ResizeObserver(() => render(last)).observe(canvas);
  return render;
}

// --- plumbing -------------------------------------------------------------------------------------------------
/** Scroll position inside a tall section while its stage is pinned, 0..1. */
function progressOf(section, navH) {
  const rect = section.getBoundingClientRect();
  const stageH = window.innerHeight - navH;
  const span = rect.height - stageH;
  return span <= 0 ? 1 : clamp01((navH - rect.top) / span);
}

export function initScenes({ meta, holdout, daily, reduced }) {
  buildHeroLine(holdout);
  const scenes = [
    { el: $("accuracy"), render: accuracyScene(meta, holdout) },
    { el: $("temperature"), render: temperatureScene(daily) },
  ];
  const bar = $("progress");
  const navH = () => document.querySelector(".nav").offsetHeight;
  let queued = false;

  const update = () => {
    queued = false;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? clamp01(window.scrollY / max) : 0})`;
    for (const s of scenes) {
      const t = reduced ? 1 : progressOf(s.el, navH());
      if (s.last === undefined || Math.abs(t - s.last) > 0.0004) {
        s.render(t);
        s.last = t;
      }
    }
  };
  const schedule = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  };
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  update();
  return { redraw: () => scenes.forEach((s) => s.render(s.last ?? 1)) };
}
