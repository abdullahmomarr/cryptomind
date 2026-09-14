#!/usr/bin/env node
// Completeness-audit checks for the CryptoMind final-report submission.
//
// One check per gate id (see GATES.md). Each prints a single success-only marker
// and exits 0 on pass, or prints a divergent line and exits 1 on fail. Pure
// reads: the filesystem and `git ls-files` only — no writes, no network.
//
//   node scripts/audit_check.mjs G1
//
import fs from "node:fs";
import { execSync } from "node:child_process";

const gate = process.argv[2];

function repoFiles() {
  // Tracked + untracked-but-not-ignored, so a report added and forgotten in the
  // working tree is still caught. .venv and site-packages are library noise.
  const tracked = execSync("git ls-files", { encoding: "utf8" });
  const untracked = execSync("git ls-files --others --exclude-standard", {
    encoding: "utf8",
  });
  return (tracked + untracked)
    .split(/\r?\n/)
    .filter((f) => f && !f.startsWith(".venv/") && !f.includes("site-packages"));
}

function pass(marker) {
  console.log(marker);
  process.exit(0);
}
function fail(msg) {
  console.log(msg);
  process.exit(1);
}

const checks = {
  // The written final report exists in no common document form. cryptomind/report.py
  // is a source module (generates figures), not the report; the how-to note is excluded.
  G1() {
    const hits = repoFiles().filter(
      (f) =>
        /(report|chapter|dissertation|final).*\.(md|docx|tex|pdf|odt)$/i.test(f) &&
        f !== "cryptomind/report.py" &&
        f !== "remove-claude-as-github-contributor.md",
    );
    return hits.length === 0 ? "REPORT_ABSENT" : "FOUND: " + hits.join(", ");
  },

  // No video file and no video script anywhere.
  G2() {
    const hits = repoFiles().filter(
      (f) =>
        /\.(mp4|mov|avi|webm|mkv)$/i.test(f) ||
        /video[-_ ]?script/i.test(f) ||
        /script.*video/i.test(f),
    );
    return hits.length === 0 ? "VIDEO_ABSENT" : "FOUND: " + hits.join(", ");
  },

  // The implementation the report must describe is present.
  G3() {
    const req = [
      "cryptomind/agent.py",
      "cryptomind/memory.py",
      "cryptomind/evaluate.py",
      "cryptomind/metrics.py",
      "cryptomind/data_layer.py",
      "cryptomind/loop.py",
      "cryptomind/seed.py",
      "cryptomind/report.py",
      "app.py",
      "run.py",
    ];
    const missing = req.filter((f) => !fs.existsSync(f));
    return missing.length === 0 ? "IMPL_PRESENT" : "MISSING: " + missing.join(", ");
  },

  // Headline evaluation results exist with the calibrated-vs-raw Brier finding + CI.
  G4() {
    const s = JSON.parse(fs.readFileSync("results/main-60d/summary.json", "utf8"));
    const p = s.pooled;
    const t = p && p.calibration_vs_raw_brier;
    const ok =
      p &&
      p.arms["memory+calibration"] &&
      p.arms["memory_raw_confidence"] &&
      t &&
      typeof t.difference === "number" &&
      typeof t.significant === "boolean" &&
      t.ci_low < t.ci_high;
    return ok
      ? "RESULTS_PRESENT n=" + p.arms["memory+calibration"].n
      : "RESULTS_INCOMPLETE";
  },

  // The Evaluation chapter's figures are rendered as non-empty PNGs.
  G5() {
    const figs = [
      "results/main-60d/figures/reliability.png",
      "results/main-60d/figures/arm_comparison.png",
      "results/main-60d/figures/confidence_shift.png",
      "results/main-60d/figures/sweep_k.png",
    ];
    const bad = figs.filter((f) => !fs.existsSync(f) || fs.statSync(f).size < 1000);
    return bad.length === 0 ? "FIGURES_PRESENT" : "MISSING/EMPTY: " + bad.join(", ");
  },

  // The LLM-backend 60-day run failed (errored per pair, no pooled result).
  G6() {
    const s = JSON.parse(
      fs.readFileSync("results/llm-groq-60d/summary.json", "utf8"),
    );
    const anyErr = Array.isArray(s.per_pair) && s.per_pair.some((p) => p.error);
    const noPooled = s.pooled === null;
    return anyErr && noPooled ? "LLM_RUN_FAILED" : "LLM_RUN_OK";
  },

  // Configured default Groq model (70B) != the model the only LLM run used (8B).
  // Independently reads both the config value and the run label, then compares.
  G7() {
    const cfg = fs.readFileSync("cryptomind/config.py", "utf8");
    // The default is the 2nd string literal on the GROQ_MODEL assignment line:
    //   GROQ_MODEL = os.environ.get("GROQ_MODEL", "<default>")
    // Match that line specifically and take the fallback literal.
    const line = cfg.match(/^GROQ_MODEL\s*=.*$/m);
    const m = line && line[0].match(/,\s*"([^"]+)"\s*\)/);
    const configured = m ? m[1] : "?";
    const smoke = JSON.parse(fs.readFileSync("results/smoke-8b/summary.json", "utf8"));
    const label = smoke.params.label;
    const mismatch = /70b/i.test(configured) && /8b/i.test(label);
    return mismatch
      ? `MODEL_MISMATCH configured=${configured} only_run=${label}`
      : `NO_MISMATCH configured=${configured} only_run=${label}`;
  },

  // A real unit-test suite exists across the assessed modules.
  G8() {
    const t = [
      "tests/test_indicators.py",
      "tests/test_memory_calibration.py",
      "tests/test_memory_outcomes.py",
      "tests/test_memory_retrieval.py",
      "tests/test_metrics.py",
      "tests/test_evaluate.py",
      "tests/test_agent_llm.py",
    ];
    const missing = t.filter((f) => !fs.existsSync(f));
    return missing.length === 0 ? "TESTS_PRESENT" : "MISSING: " + missing.join(", ");
  },

  // --- OpenRouter backend + LLM-eval gates (GATES-openrouter.md) ---

  // config.py exposes the OpenRouter base URL, the chosen model, and a key getter.
  OR1() {
    const cfg = fs.readFileSync("cryptomind/config.py", "utf8");
    const hasBase = /OPENROUTER_OPENAI_BASE_URL\s*=\s*"https:\/\/openrouter\.ai\/api\/v1"/.test(cfg);
    const hasGetter = /def get_openrouter_key\(/.test(cfg);
    const m = cfg.match(/^OPENROUTER_MODEL\s*=.*$/ms);
    const dm = m && m[0].match(/,\s*"([^"]+)"\s*\)/);
    const model = dm ? dm[1] : "?";
    return hasBase && hasGetter
      ? "OR_CONFIG_OK model=" + model
      : `OR_CONFIG_INCOMPLETE base=${hasBase} getter=${hasGetter} model=${model}`;
  },

  // agent.py builds an OpenRouter agent AND run.py accepts engine=openrouter.
  OR2() {
    const ag = fs.readFileSync("cryptomind/agent.py", "utf8");
    const hasFactory = /def for_openrouter\(/.test(ag);
    const hasBranch = /engine in \("openrouter"/.test(ag);
    const run = fs.readFileSync("run.py", "utf8");
    // openrouter must be an --engine choice in all three subcommands that take it.
    const choiceLines = (run.match(/choices=\[[^\]]*"openrouter"[^\]]*\]/g) || []).length;
    return hasFactory && hasBranch && choiceLines >= 3
      ? "OR_WIRED"
      : `OR_NOT_WIRED factory=${hasFactory} branch=${hasBranch} run_choices=${choiceLines}`;
  },

  // No hard-coded OpenRouter key (sk-or-...) in tracked source. Negative control:
  // this pattern is specific enough that a real leaked key would trip it.
  OR3() {
    const files = repoFiles().filter((f) => /\.(py|mjs|js|md|toml|txt|json|ya?ml)$/i.test(f));
    const hits = [];
    for (const f of files) {
      let txt;
      try {
        txt = fs.readFileSync(f, "utf8");
      } catch {
        continue;
      }
      // A real key literal: sk-or- followed by many key chars. The example file
      // uses the placeholder "your-openrouter-key-here", which must NOT trip.
      if (/sk-or-v1-[A-Za-z0-9]{20,}/.test(txt)) hits.push(f);
    }
    return hits.length === 0 ? "NO_HARDCODED_KEY" : "HARDCODED_KEY_IN: " + hits.join(", ");
  },

  // A completed OpenRouter walk-forward eval on disk: pooled present, no per-pair
  // errors, n>0. A re-failed run (the G6 condition) must NOT pass this.
  OR5() {
    const path = "results/llm-openrouter/summary.json";
    if (!fs.existsSync(path)) return "NO_OPENROUTER_EVAL (missing " + path + ")";
    const s = JSON.parse(fs.readFileSync(path, "utf8"));
    const perPair = Array.isArray(s.per_pair) ? s.per_pair : [];
    const anyErr = perPair.some((p) => p.error);
    const p = s.pooled;
    const n = p && p.arms && p.arms["memory+calibration"] && p.arms["memory+calibration"].n;
    if (anyErr) return "OPENROUTER_EVAL_HAS_ERRORS";
    if (!p || !n) return "OPENROUTER_EVAL_NO_POOLED";
    return `OPENROUTER_EVAL_COMPLETE pairs=${perPair.length} n=${n}`;
  },

  // The completed run recorded engine=openrouter (closes G7: the model claimed is
  // the model run). We also confirm the summary's engine label is openrouter.
  OR6() {
    const path = "results/llm-openrouter/summary.json";
    if (!fs.existsSync(path)) return "NO_OPENROUTER_EVAL";
    const s = JSON.parse(fs.readFileSync(path, "utf8"));
    const engine = s.params && s.params.engine;
    return engine === "openrouter" ? "MODEL_MATCH engine=openrouter" : "ENGINE_IS_" + engine;
  },

  // app.py wires OpenRouter into BOTH engine pickers (2 radio option strings),
  // routes it in BOTH pickers (2 uses of ENGINE_BY_LABEL feeding engine vars),
  // and detects its key (llm_key_available handles the "openrouter" engine).
  APP1() {
    const app = fs.readFileSync("app.py", "utf8");
    const options = (app.match(/"OpenRouter LLM \(free API key\)"|"OpenRouter LLM \(slower, one call per decision\)"/g) || []).length;
    const routes = (app.match(/=\s*ENGINE_BY_LABEL\(/g) || []).length;
    // key detection: the helper special-cases the openrouter engine.
    const keycheck = /engine == "openrouter"[\s\S]*?get_openrouter_key\(\)/.test(app) ? 1 : 0;
    return options === 2 && routes === 2 && keycheck === 1
      ? "APP_WIRED options=2 routes=2 keycheck=1"
      : `APP_NOT_WIRED options=${options} routes=${routes} keycheck=${keycheck}`;
  },

  // Every completed pair records n_skipped, so the skip rate is disclosed, not
  // hidden. (A "skipped" decision is one where the LLM gave no parseable reply.)
  OR7() {
    const path = "results/llm-openrouter/summary.json";
    if (!fs.existsSync(path)) return "NO_OPENROUTER_EVAL";
    const s = JSON.parse(fs.readFileSync(path, "utf8"));
    const completed = (s.per_pair || []).filter((p) => !p.error);
    if (completed.length === 0) return "NO_COMPLETED_PAIRS";
    const allHave = completed.every((p) => typeof p.n_skipped === "number");
    const total = completed.reduce((a, p) => a + (p.n_skipped || 0), 0);
    return allHave
      ? `SKIPS_RECORDED pairs=${completed.length} total_skipped=${total}`
      : "SKIPS_NOT_RECORDED";
  },
};

if (!checks[gate]) {
  fail("unknown gate: " + gate);
}
const out = checks[gate]();
// Success markers. Exact string where the marker is fixed; regex where the marker
// carries a measured value (n, pairs) that must be present but is not known ahead.
const expected = {
  G1: "REPORT_ABSENT",
  G2: "VIDEO_ABSENT",
  G3: "IMPL_PRESENT",
  G4: "RESULTS_PRESENT n=1428",
  G5: "FIGURES_PRESENT",
  G6: "LLM_RUN_FAILED",
  G7: "MODEL_MISMATCH configured=llama-3.3-70b-versatile only_run=smoke-8b",
  G8: "TESTS_PRESENT",
  OR1: "OR_CONFIG_OK model=google/gemma-4-26b-a4b-it:free",
  OR2: "OR_WIRED",
  OR3: "NO_HARDCODED_KEY",
  OR5: /^OPENROUTER_EVAL_COMPLETE pairs=\d+ n=[1-9]\d*$/,
  OR6: "MODEL_MATCH engine=openrouter",
  OR7: /^SKIPS_RECORDED pairs=[1-9]\d* total_skipped=\d+$/,
  APP1: "APP_WIRED options=2 routes=2 keycheck=1",
};
const exp = expected[gate];
const ok = exp instanceof RegExp ? exp.test(out) : out === exp;
if (ok) pass(out);
fail(out);
