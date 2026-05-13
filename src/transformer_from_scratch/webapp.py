"""Lightweight web demo.

Serves a single-page HTML UI that lets users:
  - pick a registered model version
  - enter a token prompt
  - tune temperature / top-k / max tokens
  - see generated tokens + latency

The demo can run stand-alone (``tfs-demo-web``) or alongside the API server
when the API URL is pointed at itself.
"""

from __future__ import annotations

import argparse
import logging
import os

log = logging.getLogger("tfs.webapp")

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Web demo dependencies missing. Install with: pip install 'transformer-from-scratch[api]'"
    ) from exc

# ---------------------------------------------------------------------------
# Inline HTML — no external assets, works offline
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Transformer From Scratch — Demo</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a1d2e; border-bottom: 1px solid #2d3148; padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.25rem; font-weight: 700; color: #818cf8; }
  header span { font-size: 0.75rem; color: #64748b; background: #2d3148; border-radius: 9999px; padding: 2px 8px; }
  main { flex: 1; max-width: 900px; width: 100%; margin: 2rem auto; padding: 0 1rem; }
  .card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 1rem; }
  label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.25rem; }
  input, select, textarea {
    width: 100%; background: #0f1117; border: 1px solid #2d3148; border-radius: 6px;
    color: #e2e8f0; padding: 0.5rem 0.75rem; font-size: 0.9rem;
    outline: none; transition: border-color 0.15s;
  }
  input:focus, select:focus, textarea:focus { border-color: #818cf8; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  button {
    background: #4f46e5; color: #fff; border: none; border-radius: 8px;
    padding: 0.6rem 1.5rem; font-size: 0.9rem; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
  }
  button:hover { background: #6366f1; }
  button:disabled { background: #374151; cursor: not-allowed; }
  .output-box {
    background: #0f1117; border: 1px solid #2d3148; border-radius: 8px;
    padding: 1rem; min-height: 80px; font-family: monospace; font-size: 0.85rem;
    white-space: pre-wrap; word-break: break-all; color: #a5f3fc;
  }
  .meta { display: flex; gap: 1.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
  .meta span { font-size: 0.78rem; color: #64748b; }
  .meta b { color: #94a3b8; }
  .error { color: #f87171; font-size: 0.85rem; margin-top: 0.5rem; }
  .tag { display: inline-block; background: #1e293b; border: 1px solid #2d3148; border-radius: 4px; padding: 2px 6px; font-family: monospace; font-size: 0.8rem; margin: 2px; color: #7dd3fc; }
  .tag.new { color: #86efac; background: #052e16; border-color: #166534; }
  #spinner { display: none; width: 16px; height: 16px; border: 2px solid #4f46e5; border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; margin-left: 0.5rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .score-box { font-family: monospace; font-size: 0.82rem; color: #fde68a; }
  #tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .tab { background: none; border: 1px solid #2d3148; color: #64748b; padding: 0.4rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  .tab.active { background: #2d3148; color: #e2e8f0; border-color: #4f46e5; }
  .panel { display: none; }
  .panel.active { display: block; }
  footer { text-align: center; padding: 1rem; font-size: 0.75rem; color: #374151; }
</style>
</head>
<body>
<header>
  <h1>Transformer From Scratch</h1>
  <span>v1 demo</span>
</header>
<main>
  <div id="tabs">
    <button class="tab active" onclick="switchTab('generate')">Generate</button>
    <button class="tab" onclick="switchTab('score')">Score</button>
    <button class="tab" onclick="switchTab('health')">Health</button>
  </div>

  <!-- GENERATE -->
  <div id="panel-generate" class="panel active">
    <div class="card">
      <h2>Model</h2>
      <div class="row2">
        <div>
          <label>API base URL</label>
          <input id="api-url" value="http://localhost:8000" />
        </div>
        <div>
          <label>Version</label>
          <select id="version"><option value="">— load versions —</option></select>
        </div>
      </div>
      <button onclick="loadVersions()" style="margin-top:0.25rem">Refresh Versions</button>
    </div>

    <div class="card">
      <h2>Prompt</h2>
      <label>Token IDs (comma-separated integers)</label>
      <input id="prompt-input" value="0, 1, 2" placeholder="0, 1, 2, ..." />
    </div>

    <div class="card">
      <h2>Parameters</h2>
      <div class="row">
        <div>
          <label>Max new tokens</label>
          <input type="number" id="max-new-tokens" value="16" min="1" max="512" />
        </div>
        <div>
          <label>Temperature</label>
          <input type="number" id="temperature" value="1.0" min="0.01" max="10" step="0.1" />
        </div>
        <div>
          <label>Top-k (blank = off)</label>
          <input type="number" id="top-k" value="" min="1" placeholder="disabled" />
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:0.5rem">
        <button id="gen-btn" onclick="runGenerate()">Generate</button>
        <div id="spinner"></div>
      </div>
      <div id="gen-error" class="error"></div>
    </div>

    <div class="card">
      <h2>Output</h2>
      <div class="output-box" id="gen-output">—</div>
      <div class="meta" id="gen-meta"></div>
    </div>
  </div>

  <!-- SCORE -->
  <div id="panel-score" class="panel">
    <div class="card">
      <h2>Score a sequence</h2>
      <label>Token IDs (comma-separated, min 2)</label>
      <input id="score-input" value="0, 1, 2, 3, 4" style="margin-bottom:1rem" />
      <button onclick="runScore()">Score</button>
      <div id="score-error" class="error"></div>
    </div>
    <div class="card">
      <h2>Result</h2>
      <div class="output-box score-box" id="score-output">—</div>
      <div class="meta" id="score-meta"></div>
    </div>
  </div>

  <!-- HEALTH -->
  <div id="panel-health" class="panel">
    <div class="card">
      <h2>Server health</h2>
      <button onclick="fetchHealth()">Refresh</button>
      <div class="output-box" id="health-output" style="margin-top:1rem;color:#86efac">—</div>
    </div>
    <div class="card">
      <h2>Metrics</h2>
      <button onclick="fetchMetrics()">Refresh</button>
      <div class="output-box" id="metrics-output" style="margin-top:1rem;color:#fde68a">—</div>
    </div>
  </div>
</main>
<footer>transformer-from-scratch &nbsp;·&nbsp; built from first principles</footer>

<script>
function apiBase() { return document.getElementById('api-url').value.replace(/\\/+$/, ''); }

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', ['generate','score','health'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

async function loadVersions() {
  const sel = document.getElementById('version');
  try {
    const res = await fetch(apiBase() + '/models');
    const data = await res.json();
    sel.innerHTML = data.length
      ? data.map(v => `<option value="${v.name}">${v.name}</option>`).join('')
      : '<option value="">no versions registered</option>';
  } catch(e) {
    sel.innerHTML = '<option value="">error loading versions</option>';
  }
}

function parseTokens(str) {
  return str.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
}

function renderTokens(ids, newStart) {
  return ids.map((id, i) => {
    const cls = i >= newStart ? 'tag new' : 'tag';
    return `<span class="${cls}">${id}</span>`;
  }).join('');
}

async function runGenerate() {
  const btn = document.getElementById('gen-btn');
  const spinner = document.getElementById('spinner');
  const errEl = document.getElementById('gen-error');
  errEl.textContent = '';
  btn.disabled = true; spinner.style.display = 'inline-block';

  const version = document.getElementById('version').value;
  const prompt = parseTokens(document.getElementById('prompt-input').value);
  const maxTok = parseInt(document.getElementById('max-new-tokens').value);
  const temp = parseFloat(document.getElementById('temperature').value);
  const topKRaw = document.getElementById('top-k').value.trim();
  const topK = topKRaw ? parseInt(topKRaw) : null;

  try {
    const res = await fetch(apiBase() + '/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model_version: version, prompt, max_new_tokens: maxTok,
        temperature: temp, top_k: topK
      })
    });
    const data = await res.json();
    if (!res.ok) { errEl.textContent = data.detail || JSON.stringify(data); return; }
    document.getElementById('gen-output').innerHTML = renderTokens(data.full_sequence, data.prompt.length);
    document.getElementById('gen-meta').innerHTML =
      `<span><b>Tokens generated:</b> ${data.tokens_generated}</span>` +
      `<span><b>Latency:</b> ${data.latency_ms} ms</span>` +
      `<span><b>Version:</b> ${data.model_version}</span>`;
  } catch(e) {
    errEl.textContent = 'Request failed: ' + e.message;
  } finally {
    btn.disabled = false; spinner.style.display = 'none';
  }
}

async function runScore() {
  const errEl = document.getElementById('score-error');
  errEl.textContent = '';
  const version = document.getElementById('version').value;
  const tokens = parseTokens(document.getElementById('score-input').value);
  try {
    const res = await fetch(apiBase() + '/score', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ model_version: version, tokens })
    });
    const data = await res.json();
    if (!res.ok) { errEl.textContent = data.detail || JSON.stringify(data); return; }
    document.getElementById('score-output').textContent =
      `log_probs:    [${data.log_probs.join(', ')}]\\n` +
      `mean_log_prob: ${data.mean_log_prob}\\n` +
      `perplexity:    ${data.perplexity}`;
    document.getElementById('score-meta').innerHTML =
      `<span><b>Latency:</b> ${data.latency_ms} ms</span>`;
  } catch(e) {
    errEl.textContent = 'Request failed: ' + e.message;
  }
}

async function fetchHealth() {
  try {
    const res = await fetch(apiBase() + '/health');
    const data = await res.json();
    document.getElementById('health-output').textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    document.getElementById('health-output').textContent = 'Error: ' + e.message;
  }
}

async function fetchMetrics() {
  try {
    const res = await fetch(apiBase() + '/metrics');
    const text = await res.text();
    document.getElementById('metrics-output').textContent = text;
  } catch(e) {
    document.getElementById('metrics-output').textContent = 'Error: ' + e.message;
  }
}

// Auto-load versions on page load
window.addEventListener('load', loadVersions);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

demo_app = FastAPI(title="TFS Web Demo", docs_url=None, redoc_url=None)

demo_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@demo_app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TFS Web Demo server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    uvicorn.run(
        "transformer_from_scratch.webapp:demo_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
