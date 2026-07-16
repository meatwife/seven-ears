#!/usr/bin/env python3
"""Tiny private Seven Ears dashboard.

Stdlib-only local workbench around seven_ears_card.py. Not public-facing.
Run from this directory:
    python3 dashboard_server.py --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import html
import json
import tempfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from seven_ears_card import analyze, format_card

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "private-notes" / "dashboard-uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def parse_multipart_form(body: bytes, content_type: str) -> dict[str, dict]:
    """Parse multipart/form-data with the email package (cgi is removed in Python 3.13).

    Returns {field_name: {"filename": str | None, "content": bytes}}.
    """
    msg = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii", "ignore") + b"\r\n\r\n" + body
    )
    fields: dict[str, dict] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fields[str(name)] = {
            "filename": part.get_filename(),
            "content": part.get_payload(decode=True) or b"",
        }
    return fields

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Seven Ears V1 Workbench</title>
<style>
:root { color-scheme: dark; --bg:#120914; --panel:#201226; --ink:#f7eafd; --muted:#c7a9d4; --hot:#ff7ad9; --line:#7f4a8f; --ok:#9df7c8; --warn:#ffd479; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top left, #3a1642, var(--bg) 40rem); color:var(--ink); }
main { max-width: 1080px; margin: 0 auto; padding: 2rem; }
h1 { margin: 0 0 .25rem; font-size: clamp(2rem, 5vw, 4rem); letter-spacing:-.05em; }
.tag { color:var(--muted); margin:0 0 1.5rem; }
.grid { display:grid; grid-template-columns: minmax(280px, 420px) 1fr; gap:1rem; align-items:start; }
.panel { background: color-mix(in oklab, var(--panel) 92%, black); border:1px solid #51305d; border-radius:18px; padding:1rem; box-shadow: 0 18px 60px #0008; }
label { display:block; margin:.9rem 0 .35rem; color:var(--muted); font-weight:700; font-size:.9rem; }
input, textarea, select, button { width:100%; border-radius:12px; border:1px solid #60406c; background:#130b17; color:var(--ink); padding:.75rem; font:inherit; }
textarea { min-height: 8rem; resize: vertical; }
.row { display:grid; grid-template-columns: 1fr 1fr; gap:.75rem; }
button { margin-top:1rem; background:linear-gradient(135deg, #ff7ad9, #7c5cff); border:0; color:white; font-weight:900; cursor:pointer; }
button:disabled { opacity:.55; cursor:wait; }
pre { white-space:pre-wrap; word-break:break-word; margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; line-height:1.45; }
.card { border-left:4px solid var(--hot); }
.warn { color:var(--warn); }
.ok { color:var(--ok); }
.timeline { margin: 0 0 1rem; }
.bar { position:relative; height:42px; background:#100812; border:1px solid #563163; border-radius:999px; overflow:hidden; }
.seg { position:absolute; top:8px; bottom:8px; border-radius:999px; background:linear-gradient(90deg,#ff7ad9,#ffd479); opacity:.9; }
.utt { position:absolute; top:2px; bottom:2px; border:2px solid #9df7c8; border-radius:999px; background:transparent; }
.small { color:var(--muted); font-size:.9rem; }
details { margin-top:1rem; }
summary { cursor:pointer; color:var(--muted); font-weight:800; }
@media (max-width: 820px) { .grid { grid-template-columns:1fr; } main { padding:1rem; } }
</style>
</head>
<body>
<main>
  <h1>Seven Ears</h1>
  <p class="tag">Local spoken-voice workbench. Honest ears, not a séance with ffmpeg.</p>
  <div class="grid">
    <section class="panel">
      <form id="form">
        <label>Audio file</label>
        <input name="audio" type="file" accept="audio/*,.ogg,.mp3,.wav,.m4a,.webm" required />
        <label>Transcript (optional)</label>
        <textarea name="transcript" placeholder="Leave empty to use local Whisper."></textarea>
        <div class="row">
          <div>
            <label>Transcript source</label>
            <select name="transcript_source">
              <option value="none">none</option>
              <option value="discord">discord</option>
              <option value="openclaw">openclaw</option>
              <option value="manual">manual</option>
              <option value="stt_api">stt_api</option>
              <option value="stt_whisper">stt_whisper</option>
            </select>
          </div>
          <div>
            <label>STT</label>
            <select name="stt">
              <option value="whisper" selected>local whisper</option>
              <option value="none">none</option>
            </select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Whisper model</label>
            <select name="whisper_model">
              <option value="base.en" selected>base.en</option>
              <option value="tiny.en">tiny.en</option>
              <option value="small.en">small.en</option>
            </select>
          </div>
          <div>
            <label>Whisper VAD</label>
            <select name="whisper_vad">
              <option value="true" selected>on</option>
              <option value="false">off, soft-tail test</option>
            </select>
          </div>
        </div>
        <button id="go">Analyze</button>
      </form>
      <p class="small">Audio stays local on this machine. Use 10-20 seconds of clean speech for useful profiles.</p>
    </section>
    <section class="panel card">
      <div id="status" class="small">Waiting for a voice goblin.</div>
      <div id="timeline" class="timeline"></div>
      <pre id="card"></pre>
      <details id="rawWrap" hidden><summary>Raw JSON</summary><pre id="raw"></pre></details>
    </section>
  </div>
</main>
<script>
const form = document.getElementById('form');
const go = document.getElementById('go');
const statusEl = document.getElementById('status');
const cardEl = document.getElementById('card');
const rawEl = document.getElementById('raw');
const rawWrap = document.getElementById('rawWrap');
const timelineEl = document.getElementById('timeline');
function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function renderTimeline(data){
  const dur = data.duration_s || 0;
  if (!dur) { timelineEl.innerHTML = ''; return; }
  let html = `<div class="small">Timeline: micro segments (pink/yellow), utterance spans (green outline)</div><div class="bar">`;
  for (const [s,e] of (data.active_segments || [])) {
    html += `<span class="seg" style="left:${Math.max(0,s/dur*100)}%;width:${Math.max(.6,(e-s)/dur*100)}%"></span>`;
  }
  for (const [s,e] of (data.utterance_segments || [])) {
    html += `<span class="utt" style="left:${Math.max(0,s/dur*100)}%;width:${Math.max(.6,(e-s)/dur*100)}%"></span>`;
  }
  html += `</div>`;
  timelineEl.innerHTML = html;
}
form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  go.disabled = true;
  statusEl.innerHTML = '<span class="warn">Analyzing, ear-goblin chewing...</span>';
  cardEl.textContent = '';
  timelineEl.innerHTML = '';
  rawWrap.hidden = true;
  try {
    const body = new FormData(form);
    const res = await fetch('/analyze', { method: 'POST', body });
    const payload = await res.json();
    if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${res.status}`);
    statusEl.innerHTML = '<span class="ok">Done.</span>';
    cardEl.textContent = payload.card;
    rawEl.textContent = JSON.stringify(payload.data, null, 2);
    rawWrap.hidden = false;
    renderTimeline(payload.data);
  } catch (err) {
    statusEl.innerHTML = '<span class="warn">Error: '+esc(err.message || err)+'</span>';
  } finally {
    go.disabled = false;
  }
});
</script>
</body>
</html>
"""


def parse_bool(value: str) -> bool:
    return value.lower() not in {"0", "false", "no", "off"}


class Handler(BaseHTTPRequestHandler):
    server_version = "SevenEarsWorkbench/0.1"

    def log_message(self, fmt: str, *args) -> None:  # keep terminal readable
        print(f"{self.address_string()} - {fmt % args}")

    def send_text(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_text(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_text(200, PAGE)
        else:
            self.send_text(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/analyze":
            self.send_json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(413, {"ok": False, "error": "upload missing or too large"})
            return
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self.send_json(400, {"ok": False, "error": "expected multipart/form-data"})
            return
        try:
            form = parse_multipart_form(self.rfile.read(length), ctype)
        except Exception:
            self.send_json(400, {"ok": False, "error": "could not parse multipart form"})
            return
        audio = form.get("audio")
        if audio is None or not audio["filename"]:
            self.send_json(400, {"ok": False, "error": "audio file required"})
            return

        def getfirst(name: str, default: str) -> str:
            part = form.get(name)
            return part["content"].decode("utf-8", "replace").strip() if part else default

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(audio["filename"]).suffix or ".audio"
        with tempfile.NamedTemporaryFile(prefix="seven-ears-upload-", suffix=suffix, dir=UPLOAD_DIR, delete=False) as f:
            f.write(audio["content"])
            audio_path = Path(f.name)
        try:
            transcript = getfirst("transcript", "")
            source = getfirst("transcript_source", "none")
            stt = getfirst("stt", "whisper")
            whisper_model = getfirst("whisper_model", "base.en")
            whisper_vad = parse_bool(getfirst("whisper_vad", "true"))
            data = analyze(audio_path, transcript, source, stt, whisper_model, whisper_vad)
            data["original_filename"] = html.escape(audio["filename"])
            self.send_json(200, {"ok": True, "data": data, "card": format_card(data)})
        except Exception as e:  # dashboard must fail soft
            self.send_json(500, {"ok": False, "error": str(e)})
        finally:
            # Voice notes are intimate data. The local workbench never keeps an
            # upload after analysis, including when analysis fails.
            audio_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the local Seven Ears dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Seven Ears dashboard: http://{args.host}:{args.port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
