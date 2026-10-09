// Plotly figure builders. Colours come from the CSS design tokens, so charts follow the system
// light/dark setting. Builders that animate take a progress (0..1) or opacity so app.js can
// redraw them frame by frame.

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function theme() {
  return {
    ink: cssVar("--ink"),
    muted: cssVar("--muted"),
    grid: cssVar("--grid"),
    surface: cssVar("--bg"),
    model: cssVar("--model"),
    scenario: cssVar("--scenario"),
    actual: cssVar("--actual"),
    reference: cssVar("--reference"),
    bandAlpha: parseFloat(cssVar("--band-alpha")),
    font: cssVar("--body"),
  };
}

const rgba = (hex, alpha) => {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${alpha})`;
};

const CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };
const draw = (el, data, layout) => Plotly.react(el, data, layout, CONFIG);
const axisTitle = (text) => ({ title: { text, standoff: 10, font: { size: 12, color: theme().muted } } });

function baseLayout(t, extra = {}) {
  const { xaxis = {}, yaxis = {}, legend = {}, ...rest } = extra;
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: t.font, size: 13, color: t.ink },
    margin: { l: 64, r: 14, t: 40, b: 48 },
    hoverlabel: { font: { family: t.font, size: 13 } },
    legend: { orientation: "h", x: 0, y: 1.14, traceorder: "normal", font: { size: 12, color: t.muted }, ...legend },
    xaxis: { showgrid: false, zeroline: false, automargin: true, linecolor: t.grid, tickcolor: t.grid, tickfont: { color: t.muted }, ...xaxis },
    yaxis: { gridcolor: t.grid, zeroline: false, automargin: true, tickfont: { color: t.muted }, ...yaxis },
    ...rest,
  };
}

function band(x, lower, upper, color, alpha, name, rank) {
  return [
    { x, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: rgba(color, alpha), name, hoverinfo: "skip", legendrank: rank },
  ];
}

// --- forecast ----------------------------------------------------------------------------------------
export function forecastChart(el, context, base, scenario) {
  const t = theme();
  const known = base.filter((r) => r.actual_mw !== null);
  const marker = (color) => ({ size: 8, color, line: { width: 2, color: t.surface } });
  const data = [
    ...band(base.map((r) => r.date), base.map((r) => r.lower_mw), base.map((r) => r.upper_mw), t.model, t.bandAlpha, "80% range", 3),
    {
      x: [...context.dates, ...known.map((r) => r.date)],
      y: [...context.values, ...known.map((r) => r.actual_mw)],
      type: "scatter", mode: "lines", name: "Real demand", line: { color: t.actual, width: 2 },
      hovertemplate: "%{y:,.0f} MW", legendrank: 1,
    },
    {
      x: base.map((r) => r.date), y: base.map((r) => r.predicted_mw), type: "scatter", mode: "lines+markers",
      name: "Forecast", line: { color: t.model, width: 2.5 }, marker: marker(t.model), hovertemplate: "%{y:,.0f} MW", legendrank: 2,
    },
  ];
  if (scenario) {
    data.push(
      ...band(scenario.map((r) => r.date), scenario.map((r) => r.lower_mw), scenario.map((r) => r.upper_mw), t.scenario, t.bandAlpha * 0.8, "What-if range", 5),
      {
        x: scenario.map((r) => r.date), y: scenario.map((r) => r.predicted_mw), type: "scatter", mode: "lines+markers",
        name: "What-if forecast", line: { color: t.scenario, width: 2.5 }, marker: marker(t.scenario),
        hovertemplate: "%{y:,.0f} MW", legendrank: 4,
      },
    );
  }
  const origin = context.dates[context.dates.length - 1];
  const narrow = el.clientWidth < 560; // the legend wraps to two rows on phones
  draw(el, data, baseLayout(t, {
    hovermode: "x unified",
    margin: { l: 72, r: 14, t: narrow ? 78 : 44, b: 48 },
    legend: { y: narrow ? 1.28 : 1.14 },
    xaxis: { type: "date", tickformat: "%b %-d" },
    yaxis: { ...axisTitle("Daily mean demand (MW)"), tickformat: ",.0f" },
    shapes: [{ type: "line", x0: origin, x1: origin, yref: "paper", y0: 0, y1: 1, line: { color: t.reference, width: 1, dash: "dot" } }],
    annotations: [{ x: origin, xshift: -4, yref: "paper", y: 1, text: "forecast made", showarrow: false, xanchor: "right", yanchor: "top", font: { size: 11, color: t.muted } }],
  }));
}

// --- accuracy -------------------------------------------------------------------------------------------
export function compareChart(el, comparison, finalName, progress = 1) {
  const t = theme();
  const rows = [...comparison].sort((a, b) => b.mape - a.mape); // best ends up on top
  draw(el, [{
    type: "bar", orientation: "h", x: rows.map((r) => r.mape * progress), y: rows.map((r) => r.model),
    marker: { color: rows.map((r) => (r.model === finalName ? t.model : t.reference)) },
    text: rows.map((r) => `${(r.mape * progress).toFixed(1)}%`), textposition: "outside", cliponaxis: false,
    textfont: { color: t.ink, size: 13 }, hovertemplate: "%{y}<br>%{x:.2f}%<extra></extra>",
  }], baseLayout(t, {
    margin: { l: 10, r: 46, t: 8, b: 40 }, bargap: 0.36, showlegend: false,
    xaxis: { range: [0, Math.max(...rows.map((r) => r.mape)) * 1.12], showticklabels: false, showline: false, ticks: "" },
    yaxis: { showgrid: false, automargin: true, ticksuffix: "  " },
  }));
}

export function horizonChart(el, horizon, progress = 1) {
  const t = theme();
  const pick = (weather) => horizon.filter((r) => r.weather === weather).sort((a, b) => a.horizon - b.horizon);
  const shown = (rows) => rows.slice(0, Math.max(1, Math.ceil(rows.length * progress)));
  const known = pick("actual");
  const normal = pick("climatology");
  const line = (rows, key, name, color, width) => {
    const r = shown(rows);
    return { x: r.map((p) => p.horizon), y: r.map((p) => p[key]), type: "scatter", mode: "lines+markers", name, line: { color, width }, marker: { size: 8 }, hovertemplate: "%{y:.1f}%" };
  };
  draw(el, [
    line(known, "seasonal_naive_mape", "Last week's value", t.reference, 2),
    line(normal, "mape", "Model, normal weather", t.scenario, 2),
    line(known, "mape", "Model, weather known", t.model, 2.5),
  ], baseLayout(t, {
    hovermode: "x unified", margin: { l: 56, r: 14, t: 56, b: 48 }, legend: { y: 1.22 },
    xaxis: { ...axisTitle("Days ahead"), dtick: 1, range: [0.6, 7.4] },
    yaxis: { ...axisTitle("Error (%)"), range: [0, 11] },
  }));
}

// --- temperature ----------------------------------------------------------------------------------------------
export function scatterChart(el, dates, temps, demand, opacity = 0.45) {
  const t = theme();
  const weekend = dates.map((d) => new Date(`${d}T00:00:00Z`).getUTCDay() % 6 === 0); // Sun(0) or Sat(6)
  const trace = (flag, name, color) => {
    const idx = weekend.map((w, i) => (w === flag ? i : -1)).filter((i) => i >= 0);
    return {
      x: idx.map((i) => temps[i]), y: idx.map((i) => demand[i]), type: "scatter", mode: "markers", name,
      marker: { size: 4, color: rgba(color, opacity) }, hovertemplate: "%{x:.1f} °C: %{y:,.0f} MW",
    };
  };
  draw(el, [trace(false, "Weekdays", t.model), trace(true, "Weekends", t.scenario)], baseLayout(t, {
    legend: { itemsizing: "constant" },
    xaxis: axisTitle("Daily mean temperature (°C)"),
    yaxis: { ...axisTitle("Daily mean demand (MW)"), tickformat: ",.0f" },
  }));
}
