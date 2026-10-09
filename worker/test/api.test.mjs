// Runs the real Worker handler against the real exported data, with a stub ASSETS binding.
//   node --test worker/test/
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import worker from "../index.js";

const WEB = new URL("../../web/", import.meta.url);
const env = {
  ASSETS: {
    async fetch(url) {
      try {
        const body = await readFile(new URL(`.${new URL(url).pathname}`, WEB));
        return new Response(body, { headers: { "content-type": "application/json" } });
      } catch {
        return new Response("not found", { status: 404 });
      }
    },
  },
};

const call = async (path, init) => {
  const res = await worker.fetch(new Request(`https://example.test${path}`, init), env);
  return { status: res.status, body: await res.json(), headers: res.headers };
};

test("health reports the data window", async () => {
  const { status, body } = await call("/api/health");
  assert.equal(status, 200);
  assert.equal(body.status, "ok");
  assert.equal(body.data_through, "2018-08-02");
});

test("model card includes the comparison table", async () => {
  const { body } = await call("/api/model");
  assert.ok(body.holdout.mape < body.baseline.mape);
  assert.ok(body.comparison.length >= 6);
});

test("forecast returns ordered intervals and is cacheable", async () => {
  const { status, body, headers } = await call("/api/forecast?origin=2017-07-10&horizon=3&temp_shift_c=2");
  assert.equal(status, 200);
  assert.equal(body.points.length, 3);
  for (const p of body.points) assert.ok(p.lower_mw < p.predicted_mw && p.predicted_mw < p.upper_mw);
  assert.match(headers.get("cache-control"), /max-age/);
  assert.equal(headers.get("access-control-allow-origin"), "*");
});

test("forecast beyond the data has null actuals", async () => {
  const { body } = await call("/api/forecast?origin=2018-08-02&horizon=2");
  assert.ok(body.points.every((p) => p.actual_mw === null && p.weather_source === "climatology"));
});

test("a heatwave raises summer demand", async () => {
  const base = (await call("/api/forecast?origin=2017-07-10&horizon=5")).body.points;
  const hot = (await call("/api/forecast?origin=2017-07-10&horizon=5&temp_shift_c=5")).body.points;
  assert.ok(hot.every((p, i) => p.predicted_mw > base[i].predicted_mw));
});

for (const query of [
  "origin=1990-01-01",
  "origin=2017-07-10&horizon=99",
  "origin=2017-07-10&horizon=2.5",
  "origin=2017-07-10&weather=sunny",
  "origin=2017-07-10&temp_shift_c=100",
  "origin=not-a-date",
  "",
]) {
  test(`forecast rejects bad query '${query}'`, async () => {
    const { status, body } = await call(`/api/forecast?${query}`);
    assert.equal(status, 422);
    assert.ok(body.detail);
  });
}

const history = Array.from({ length: 28 }, (_, i) => 38_000 + 100 * i);
const post = (payload) => call("/api/predict", { method: "POST", body: JSON.stringify(payload) });

test("predict returns a day-ahead value with an interval", async () => {
  const { status, body } = await post({ date: "2017-07-11", recent_demand_mw: history, temp_mean: 27, temp_max: 32, temp_min: 22 });
  assert.equal(status, 200);
  assert.ok(body.lower_mw < body.predicted_mw && body.predicted_mw < body.upper_mw);
});

test("predict validates its body", async () => {
  const base = { date: "2017-07-11", temp_mean: 27 };
  assert.equal((await post({ ...base, recent_demand_mw: [30_000] })).status, 422);
  assert.equal((await post({ ...base, recent_demand_mw: Array(28).fill(1) })).status, 422);
  assert.equal((await post({ date: "2017-07-11", recent_demand_mw: history })).status, 422);
  const bad = await call("/api/predict", { method: "POST", body: "{nope" });
  assert.equal(bad.status, 422);
});

test("unknown routes and wrong methods", async () => {
  assert.equal((await call("/api/nope")).status, 404);
  assert.equal((await call("/api/predict")).status, 405);
});
