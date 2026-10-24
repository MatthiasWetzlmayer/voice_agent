"""Schreibgeschützter Datenendpunkt für das Dashboard in ./ui.

GET /api/state liefert den aktuellen Zustand als JSON. Läuft in einem eigenen Thread mit der
Standardbibliothek und hört nur auf localhost. Der Zustand wird im Event-Loop des Agenten
gelesen, damit nie halb geänderte Daten gesendet werden.
"""

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("dashboard")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/api/state":
            self.send_error(404)
            return
        future = asyncio.run_coroutine_threadsafe(self.server.read_state(), self.server.loop)
        body = json.dumps(future.result(timeout=2), ensure_ascii=False, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")  # die UI läuft auf einem anderen Port
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keine Zeile pro Abfrage
        pass


def start_dashboard_api(assistant, info: dict, port: int) -> ThreadingHTTPServer | None:
    """Startet den Endpunkt. `info` landet unverändert unter "agent" in der Antwort."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError as e:
        log.warning("Dashboard-Endpunkt nicht gestartet (Port %d): %s", port, e)
        return None
    agent = {**info, "started": time.time()}

    async def read_state() -> dict:
        return {"agent": agent, **assistant.snapshot()}

    server.loop, server.read_state = asyncio.get_running_loop(), read_state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Daten fürs Dashboard: http://127.0.0.1:%d/api/state", port)
    return server
