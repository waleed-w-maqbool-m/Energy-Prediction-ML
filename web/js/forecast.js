// JavaScript port of energy_forecast.features + energy_forecast.forecast.
//
// A dependency-free ES module that runs unchanged in the browser (dashboard) and in a
// Cloudflare Worker (API). It evaluates the exported gradient-boosting trees and reproduces
// the Python feature engineering and recursive forecasting; tests/test_js_parity.py checks
// that both implementations agree to floating-point precision.

const DAY_MS = 86_400_000;

// --- dates as integer day numbers (days since 1970-01-01, UTC) -------------------------
export const dayNumber = (iso) => {
  const [y, m, d] = iso.split("-").map(Number);
  return Date.UTC(y, m - 1, d) / DAY_MS;
};
export const isoFromDay = (n) => new Date(n * DAY_MS).toISOString().slice(0, 10);
const weekday = (n) => (((n + 3) % 7) + 7) % 7; // Monday = 0 (1970-01-01 was a Thursday)
const monthOf = (n) => new Date(n * DAY_MS).getUTCMonth() + 1;
const dayOfYear = (n) => {
  const year = new Date(n * DAY_MS).getUTCFullYear();
  return n - Date.UTC(year, 0, 1) / DAY_MS + 1;
};

// Must match SIN_COS_DECIMALS / ROLL_*_DECIMALS in energy_forecast/features.py.
const roundTo = (x, decimals) => {
  const k = 10 ** decimals;
  return Math.round(x * k) / k;
};

const mean = (arr, from, to) => {
  let s = 0;
  for (let i = from; i < to; i++) s += arr[i];
  return s / (to - from);
};

// --- the model: sum of regression trees ---------------------------------------------------
export class TreeModel {
  constructor({ features, baseline, trees }) {
    this.features = features;
    this.baseline = baseline;
    this.trees = trees;
  }

  /** Predict one row. `x` is ordered like `this.features`. */
  predict(x) {
    let total = this.baseline;
    for (const tree of this.trees) {
      let node = 0;
      while (tree.f[node] >= 0) {
        node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
      }
      total += tree.v[node];
    }
    return total;
  }
}

// --- feature engineering (mirrors energy_forecast/features.py) -------------------------------
/**
 * Features for row `i` of a consecutive daily series.
 * `ctx`: { day0, demand, tmean, tmax, tmin } where index 0 is day number `day0`.
 * Demand features only look at rows before `i`.
 */
function featureRow(ctx, i, holidays, balanceTemp, order) {
  const n = ctx.day0 + i;
  const dow = weekday(n);
  const doy = dayOfYear(n);
  const angle = (2 * Math.PI * doy) / 365.25;
  const isNonworking = (day) => weekday(day) >= 5 || holidays.has(day);
  const tm = ctx.tmean[i];
  const y = ctx.demand;

  const f = {
    dow,
    month: monthOf(n),
    doy_sin: roundTo(Math.sin(angle), 10),
    doy_cos: roundTo(Math.cos(angle), 10),
    is_weekend: dow >= 5 ? 1 : 0,
    is_holiday: holidays.has(n) ? 1 : 0,
    pre_holiday: holidays.has(n + 1) ? 1 : 0,
    post_holiday: holidays.has(n - 1) ? 1 : 0,
    is_nonworking: isNonworking(n) ? 1 : 0,
    temp_mean: tm,
    temp_max: ctx.tmax[i],
    temp_min: ctx.tmin[i],
    hdd: Math.max(balanceTemp - tm, 0),
    cdd: Math.max(tm - balanceTemp, 0),
    temp_mean_roll3: roundTo(mean(ctx.tmean, i - 2, i + 1), 8),
    temp_delta_1: tm - ctx.tmean[i - 1],
    lag_1: y[i - 1],
    lag_2: y[i - 2],
    lag_7: y[i - 7],
    lag_14: y[i - 14],
    roll7_mean: roundTo(mean(y, i - 7, i), 6),
    roll28_mean: roundTo(mean(y, i - 28, i), 6),
    lag_1_nonworking: isNonworking(n - 1) ? 1 : 0,
    lag_7_nonworking: isNonworking(n - 7) ? 1 : 0,
  };
  return order.map((name) => {
    if (!(name in f)) throw new Error(`model expects unknown feature '${name}'`);
    return f[name];
  });
}

// --- the forecaster --------------------------------------------------------------------------------
export class Forecaster {
  /**
   * @param {{model: object, meta: object, daily: object}} data  contents of model.json, meta.json, daily.json
   */
  constructor({ model, meta, daily }) {
    this.model = new TreeModel(model);
    this.meta = meta;
    this.c = meta.constants;
    this.holidays = new Set(meta.holidays.map(dayNumber));
    this.day0 = dayNumber(daily.start);
    this.demand = Float64Array.from(daily.demand_mw);
    this.tmean = Float64Array.from(daily.temp_mean);
    this.tmax = Float64Array.from(daily.temp_max);
    this.tmin = Float64Array.from(daily.temp_min);
    this.size = this.demand.length;
    this.firstOrigin = this.day0 + this.c.warmup_days - 1;
    this.lastDay = this.day0 + this.size - 1;
    this.clim = meta.climatology;
  }

  get firstOriginISO() { return isoFromDay(this.firstOrigin); }
  get lastDateISO() { return isoFromDay(this.lastDay); }

  /** Weather for one day: observed when allowed and available, otherwise climatology. */
  _weather(day, mode, shift) {
    const idx = day - this.day0;
    const observed = mode === "actual" && idx >= 0 && idx < this.size;
    let w;
    if (observed) {
      w = [this.tmean[idx], this.tmax[idx], this.tmin[idx]];
    } else {
      const k = Math.min(dayOfYear(day), 365) - 1;
      w = [this.clim.temp_mean[k], this.clim.temp_max[k], this.clim.temp_min[k]];
    }
    return { tmean: w[0] + shift, tmax: w[1] + shift, tmin: w[2] + shift, source: observed ? "observed" : "climatology" };
  }

  /** Recursive forecast: each prediction becomes the next day's lag input. */
  _recurse(ctx, start, horizon) {
    const out = new Array(horizon);
    for (let s = 0; s < horizon; s++) {
      const x = featureRow(ctx, start + s, this.holidays, this.c.balance_temp_c, this.model.features);
      out[s] = this.model.predict(x);
      ctx.demand[start + s] = out[s];
    }
    return out;
  }

  /**
   * Forecast `horizon` days after `origin` (the last day of known demand).
   * @param {{origin: string, horizon?: number, weather?: "actual"|"climatology", tempShift?: number}} opts
   */
  forecast({ origin, horizon = 7, weather = "actual", tempShift = 0 }) {
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > this.c.max_horizon) {
      throw new RangeError(`horizon must be between 1 and ${this.c.max_horizon}`);
    }
    if (weather !== "actual" && weather !== "climatology") {
      throw new RangeError("weather must be 'actual' or 'climatology'");
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(origin ?? "")) throw new RangeError("origin must be a YYYY-MM-DD date");
    const originDay = dayNumber(origin);
    if (originDay < this.firstOrigin || originDay > this.lastDay) {
      throw new RangeError(`origin must be between ${this.firstOriginISO} and ${this.lastDateISO}`);
    }

    // Window = last `warmup` days of history, then the days to forecast.
    const warm = this.c.warmup_days;
    const from = originDay - this.day0 - warm + 1;
    const total = warm + horizon;
    const ctx = {
      day0: originDay - warm + 1,
      demand: new Float64Array(total).fill(NaN),
      tmean: new Float64Array(total),
      tmax: new Float64Array(total),
      tmin: new Float64Array(total),
    };
    for (let k = 0; k < warm; k++) {
      ctx.demand[k] = this.demand[from + k];
      ctx.tmean[k] = this.tmean[from + k];
      ctx.tmax[k] = this.tmax[from + k];
      ctx.tmin[k] = this.tmin[from + k];
    }
    const wx = [];
    for (let s = 0; s < horizon; s++) {
      const w = this._weather(originDay + 1 + s, weather, tempShift);
      wx.push(w);
      ctx.tmean[warm + s] = w.tmean;
      ctx.tmax[warm + s] = w.tmax;
      ctx.tmin[warm + s] = w.tmin;
    }

    const pred = this._recurse(ctx, warm, horizon);
    const bands = this.meta.intervals[weather];
    return pred.map((p, s) => {
      const day = originDay + 1 + s;
      const [lo, hi] = bands[String(s + 1)];
      const idx = day - this.day0;
      return {
        date: isoFromDay(day),
        predicted_mw: p,
        lower_mw: p * (1 + lo),
        upper_mw: p * (1 + hi),
        actual_mw: idx >= 0 && idx < this.size ? this.demand[idx] : null,
        temp_mean_c: wx[s].tmean,
        weather_source: wx[s].source,
      };
    });
  }

  /**
   * One-day-ahead prediction from caller-supplied history (used by POST /predict).
   * `recentDemand` are consecutive daily means ending the day before `date`; at least 28 values.
   */
  predictSingle({ date, recentDemand, tempMean, tempMax = null, tempMin = null, prevTempMean = null }) {
    const warm = this.c.warmup_days;
    if (!Array.isArray(recentDemand) || recentDemand.length < warm) {
      throw new RangeError(`recent_demand_mw needs at least ${warm} daily values`);
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date ?? "")) throw new RangeError("date must be a YYYY-MM-DD date");
    const prev = prevTempMean ?? tempMean;
    const total = warm + 1;
    const ctx = {
      day0: dayNumber(date) - warm,
      demand: Float64Array.from([...recentDemand.slice(-warm), NaN]),
      tmean: new Float64Array(total).fill(prev),
      tmax: new Float64Array(total).fill(prev),
      tmin: new Float64Array(total).fill(prev),
    };
    ctx.tmean[warm] = tempMean;
    ctx.tmax[warm] = tempMax ?? tempMean;
    ctx.tmin[warm] = tempMin ?? tempMean;
    const [pred] = this._recurse(ctx, warm, 1);
    const [lo, hi] = this.meta.intervals.actual["1"];
    return { date, predicted_mw: pred, lower_mw: pred * (1 + lo), upper_mw: pred * (1 + hi) };
  }
}
