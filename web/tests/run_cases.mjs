// Reads a JSON array of forecast cases on stdin, evaluates them with the JavaScript
// forecaster and the exported model, and writes the results to stdout.
// Used by tests/test_js_parity.py to compare against the Python implementation.
import { readFileSync } from "node:fs";
import { Forecaster } from "../js/forecast.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
const forecaster = new Forecaster({ model: load("model.json"), meta: load("meta.json"), daily: load("daily.json") });

const cases = JSON.parse(readFileSync(0, "utf8"));
const results = cases.map((c) => {
  try {
    return c.kind === "single"
      ? forecaster.predictSingle(c.args)
      : forecaster.forecast(c.args);
  } catch (err) {
    return { error: err.message };
  }
});
process.stdout.write(JSON.stringify(results));
