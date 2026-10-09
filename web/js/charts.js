// The interactive forecast chart (Plotly) and the shared theme helpers. Colours come from the CSS
// design tokens, so charts follow the system light/dark setting.

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

export const rgba = (hex, alpha) => {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${alpha})`;
};

function band(x, lower, upper, color, alpha, name, rank) {
  return [
    { x, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: rgba(color, alpha), name, hoverinfo: "skip", legendrank: rank },
  ];
}

/** Recent real demand, the 7-day forecast with its range, what happened next, and an optional what-if. */
export function forecastChart(el, context, base, scenario) {
  const t = theme();
  const known = base.filter((r) => r.actual_mw !== null);
  const marker = (color) => ({ size: 8, color, line: { width: 2, color: t.surface } });
  const data = [
    ...band(base.map((r) => r.date), base.map((r) => r.lower_mw), base.map((r) => r.upper_mw), t.model, t.bandAlpha, "Range", 3),
    {
      x: [...context.dates, ...known.map((r) => r.date)],
      y: [...context.values, ...known.map((r) => r.actual_mw)],
      type: "scatter", mode: "lines", name: "Real", line: { color: t.actual, width: 2 },
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
        name: "What if", line: { color: t.scenario, width: 2.5 }, marker: marker(t.scenario), hovertemplate: "%{y:,.0f} MW", legendrank: 4,
      },
    );
  }
  const origin = context.dates[context.dates.length - 1];
  const narrow = el.clientWidth < 560; // the legend wraps to two rows on phones
  Plotly.react(el, data, {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: t.font, size: 13, color: t.ink },
    margin: { l: 66, r: 14, t: narrow ? 70 : 40, b: 40 },
    hovermode: "x unified",
    hoverlabel: { font: { family: t.font, size: 13 } },
    legend: { orientation: "h", x: 0, y: narrow ? 1.26 : 1.13, traceorder: "normal", font: { size: 12, color: t.muted } },
    xaxis: { type: "date", tickformat: "%b %-d", showgrid: false, zeroline: false, automargin: true, linecolor: t.grid, tickcolor: t.grid, tickfont: { color: t.muted } },
    yaxis: { tickformat: ",.0f", ticksuffix: " MW", gridcolor: t.grid, zeroline: false, automargin: true, tickfont: { color: t.muted } },
    shapes: [{ type: "line", x0: origin, x1: origin, yref: "paper", y0: 0, y1: 1, line: { color: t.reference, width: 1, dash: "dot" } }],
    annotations: [{ x: origin, xshift: -4, yref: "paper", y: 1, text: "forecast made", showarrow: false, xanchor: "right", yanchor: "top", font: { size: 11, color: t.muted } }],
  }, { displayModeBar: false, responsive: true });
}
