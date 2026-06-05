"""
MothServer — Groovy pSync output node
======================================
Serves the Mellow Moth web page and pushes the live group-synchrony value to
the browser, so the moth flies on real data — with no separate bridge process
and no edits to the moth HTML file.

    GroupSyncScore.scores ──▶ MothServer.data
    then open  http://localhost:<port>

How it works
------------
- setup() starts a tiny HTTP server on a background daemon thread (the same
  pattern MidiIn uses for its MIDI listener thread).
- process(data) reads one column from the incoming `scores` table (default
  'frac_in_sync', range 0..1) and stores it as the latest value.
- The browser connects to /events and receives that value via Server-Sent
  Events — a push callback, not polling. The moth page is served with a small
  script injected just before </body>, so the HTML file itself stays untouched.
- terminate() shuts the server down (how MidiIn closes its ports).

This is an input-only "sink" node, exactly like OSCOut.

Parameters
----------
server / port          HTTP port. Open http://localhost:<port>. Default 8000.
server / html_path     Path to the moth HTML. Blank = the bundled
                       groovy_moth_visualizer.html at the repo root.
server / value_key     Column from the scores table that drives the moth
                       (0..1). Default 'frac_in_sync'.
server / open_browser  Toggle True to open the moth page in your browser.
"""

import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


# Injected just before </body> when the page is served. Drives the existing
# slider from the live value, so the moth HTML never has to be edited.
_INJECT = """
<script>
/* injected by goofi MothServer -- drive the slider from live group synchrony */
(function () {
  var slider = document.getElementById("syncSlider");
  if (!slider) return;
  var target = null;                  // null until the first live value arrives
  try {
    var es = new EventSource("/events");
    es.onmessage = function (e) {
      try { target = Math.max(0, Math.min(1, Number(JSON.parse(e.data).sync))); }
      catch (_) {}
    };
  } catch (_) {}
  setInterval(function () {
    if (target === null) return;      // no live data -> manual slider still works
    var cur = Number(slider.value);
    slider.value = cur + (target - cur) * 0.15;   // smooth ease toward live value
  }, 33);
})();
</script>
"""


def _fallback_page(msg):
    return (
        "<!DOCTYPE html><html><body style='font-family:sans-serif;"
        "background:#1d1b22;color:#f4e7d0;padding:40px'>"
        f"<h2>MothServer</h2><p>{msg}</p></body></html>"
    ).encode()


class MothServer(Node):
    """Serve the moth page and push live group synchrony to it via SSE."""

    NO_MULTIPROCESSING = True  # background server thread — run in the main process

    @staticmethod
    def config_input_slots():
        return {"data": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "server": {
                "port": IntParam(
                    8000, 1024, 65535,
                    doc="HTTP port. Open http://localhost:<port> in a browser."),
                "html_path": StringParam(
                    "", doc="Path to the moth HTML. Blank = the bundled "
                            "groovy_moth_visualizer.html at the repo root."),
                "value_key": StringParam(
                    "frac_in_sync",
                    doc="Column from the scores table that drives the moth (0..1)."),
                "open_browser": BoolParam(
                    False, doc="Toggle True to open the moth page in your browser."),
            }
        }

    # ── lifecycle ──────────────────────────────────────────────────────
    def setup(self):
        self._lock = threading.Lock()
        self._latest = {"sync": 0.0, "n_in_sync": 0.0, "n_total": 0.0, "phase": ""}
        self._stop = threading.Event()
        self._httpd = None
        self._start_server()

    def terminate(self):
        self._stop_server()

    # ── server management ──────────────────────────────────────────────
    def _resolve_html(self):
        p = self.params.server.html_path.value.strip()
        if p:
            return p
        try:
            # src/goofi/nodes/outputs/mothserver.py -> repo root is 4 levels up
            return str(Path(__file__).resolve().parents[4] / "groovy_moth_visualizer.html")
        except Exception:
            return ""

    def _start_server(self):
        port = int(self.params.server.port.value)
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # keep goofi's console quiet

            def _headers(self, code, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

            def do_GET(self):
                # ── Server-Sent Events: push the latest value (the callback) ──
                if self.path.startswith("/events"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    try:
                        while not node._stop.is_set():
                            with node._lock:
                                payload = json.dumps(node._latest)
                            self.wfile.write(f"data: {payload}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(0.1)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                # ── JSON snapshot (polling fallback / debugging) ──
                if self.path.startswith("/score"):
                    with node._lock:
                        body = json.dumps(node._latest).encode()
                    self._headers(200, "application/json")
                    self.wfile.write(body)
                    return

                # ── the moth page, with the SSE snippet injected ──
                html_path = node._resolve_html()
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    idx = html.lower().rfind("</body>")
                    html = (html[:idx] + _INJECT + html[idx:]) if idx != -1 else (html + _INJECT)
                    self._headers(200, "text/html; charset=utf-8")
                    self.wfile.write(html.encode("utf-8"))
                except OSError:
                    self._headers(404, "text/html; charset=utf-8")
                    self.wfile.write(_fallback_page(
                        f"Could not read moth HTML at '{html_path}'. "
                        "Set the server/html_path parameter."))

        self._stop.clear()
        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as e:
            print(f"[MothServer] could not bind port {port}: {e}")
            self._httpd = None
            return
        self._httpd.daemon_threads = True
        threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="MothServer-http"
        ).start()
        print(f"[MothServer] serving moth on http://localhost:{port}  "
              f"(wire GroupSyncScore.scores -> data)")

    def _stop_server(self):
        self._stop.set()
        httpd = getattr(self, "_httpd", None)
        if httpd is not None:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    # ── parameter callbacks ────────────────────────────────────────────
    def server_port_changed(self, value):
        if not hasattr(self, "_stop"):
            return  # callback fired before setup() — ignore
        self._stop_server()
        self._start_server()

    def server_open_browser_changed(self, value):
        if value:
            port = int(self.params.server.port.value)
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception as e:
                print(f"[MothServer] could not open browser: {e}")

    # ── process ────────────────────────────────────────────────────────
    def process(self, data):
        if data is None:
            return None
        table = data.data

        def scalar(key, default=float("nan")):
            try:
                return float(table[key].data[0])
            except Exception:
                return default

        def string(key, default=""):
            try:
                v = table[key].data
                return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception:
                return default

        sync = scalar(self.params.server.value_key.value)
        if math.isnan(sync):
            sync = scalar("frac_in_sync", 0.0)
        sync = max(0.0, min(1.0, sync))

        with self._lock:
            self._latest = {
                "sync": sync,
                "n_in_sync": scalar("n_in_sync", 0.0),
                "n_total": scalar("n_total", 0.0),
                "phase": string("phase"),
            }
        return None
