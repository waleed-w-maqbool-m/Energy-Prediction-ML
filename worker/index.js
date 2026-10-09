// Cloudflare Worker: the forecast API.
//
// The dashboard (web/) is served as static assets; this Worker handles /api/* only
// (see `run_worker_first` in wrangler.jsonc). It evaluates the same exported model with the
// same JavaScript forecaster the browser uses, so the API and the dashboard cannot disagree.
//
//   GET  /api/health
//   GET  /api/model
//   GET  /api/forecast?origin=2017-07-10&horizon=7&weather=actual&temp_shift_c=0
//   POST /api/predict   { date, recent_demand_mw: [28..60 values], temp_mean, temp_max?, temp_min?, prev_temp_mean? }

import { Forecaster } from "../web/js/forecast.js";

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "access-control-allow-origin": "*",
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { ...JSON_HEADERS, ...extra } });

// The model and data are static assets; load them once per isolate.
let loading;
function getForecaster(request, env) {
  loading ??= (async () => {
    const get = async (name) => {
      const res = await env.ASSETS.fetch(new URL(`/data/${name}`, request.url));
      if (!res.ok) throw new Error(`could not load ${name} (${res.status})`);
      return res.json();
    };
    const [model, meta, daily] = await Promise.all([get("model.json"), get("meta.json"), get("daily.json")]);
    return new Forecaster({ model, meta, daily });
  })().catch((err) => {
    loading = undefined; // allow a retry on the next request
    throw err;
  });
  return loading;
}

// --- input parsing -----------------------------------------------------------------------------------
function numberParam(params, name, { fallback, min, max, integer = false }) {
  const raw = params.get(name);
  if (raw === null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new HttpError(422, `${name} must be ${integer ? "an integer" : "a number"}`);
  }
  if (value < min || value > max) throw new HttpError(422, `${name} must be between ${min} and ${max}`);
  return value;
}

function optionalNumber(body, name, min, max) {
  const value = body[name];
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new HttpError(422, `${name} must be a number between ${min} and ${max}`);
  }
  return value;
}

const shapeForecast = (rows) => rows.map((r) => ({ ...r }));

// --- routes ------------------------------------------------------------------------------------------------
const routes = {
  "GET /api/health": async (_req, f) => ({
    status: "ok",
    data_through: f.lastDateISO,
    trained_at: f.meta.trained_at,
  }),

  "GET /api/model": async (_req, f) => {
    const { model, split, holdout, baseline, improvement_pct, comparison } = f.meta;
    return { model, split, holdout, baseline, improvement_pct, comparison };
  },

  "GET /api/forecast": async (req, f) => {
    const params = new URL(req.url).searchParams;
    const origin = params.get("origin");
    if (!origin) throw new HttpError(422, "origin is required (YYYY-MM-DD)");
    const horizon = numberParam(params, "horizon", { fallback: 7, min: 1, max: f.c.max_horizon, integer: true });
    const tempShift = numberParam(params, "temp_shift_c", { fallback: 0, min: -15, max: 15 });
    const weather = params.get("weather") ?? "actual";
    try {
      const points = f.forecast({ origin, horizon, weather, tempShift });
      return { origin, horizon, weather, temp_shift_c: tempShift, points: shapeForecast(points) };
    } catch (err) {
      throw new HttpError(422, err.message);
    }
  },

  "POST /api/predict": async (req, f) => {
    let body;
    try {
      body = await req.json();
    } catch {
      throw new HttpError(422, "request body must be valid JSON");
    }
    const recent = body?.recent_demand_mw;
    if (!Array.isArray(recent) || recent.length < 28 || recent.length > 60) {
      throw new HttpError(422, "recent_demand_mw must be an array of 28 to 60 daily values");
    }
    if (!recent.every((v) => typeof v === "number" && v >= 5_000 && v <= 100_000)) {
      throw new HttpError(422, "recent_demand_mw values must be daily means between 5,000 and 100,000 MW");
    }
    const tempMean = optionalNumber(body, "temp_mean", -40, 50);
    if (tempMean === null) throw new HttpError(422, "temp_mean is required (deg C)");
    try {
      return f.predictSingle({
        date: body.date,
        recentDemand: recent,
        tempMean,
        tempMax: optionalNumber(body, "temp_max", -40, 55),
        tempMin: optionalNumber(body, "temp_min", -50, 50),
        prevTempMean: optionalNumber(body, "prev_temp_mean", -40, 50),
      });
    } catch (err) {
      throw new HttpError(422, err.message);
    }
  },
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }

    const handler = routes[`${request.method} ${url.pathname}`];
    if (!handler) {
      const known = Object.keys(routes).some((k) => k.endsWith(` ${url.pathname}`));
      return json({ detail: known ? "method not allowed" : "not found" }, known ? 405 : 404);
    }
    try {
      const forecaster = await getForecaster(request, env);
      const body = await handler(request, forecaster);
      // Forecasts are deterministic for a given URL, so they are safe to cache.
      const cache = request.method === "GET" ? { "cache-control": "public, max-age=3600" } : {};
      return json(body, 200, cache);
    } catch (err) {
      if (err instanceof HttpError) return json({ detail: err.message }, err.status);
      console.error(err);
      return json({ detail: "internal error" }, 500);
    }
  },
};
