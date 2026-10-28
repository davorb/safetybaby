"""A stand-in for the babymon service so the TV app can be exercised without a camera.

    python3 tvos/scripts/mock_server.py FRAME.jpg [--port 8080]
    curl localhost:8080/mock?scenario=calm|pending|high|emergency|offline|starting

Serves the same JSON routes as babymon/web.py with canned data.
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

FRAME = open(sys.argv[1], "rb").read()
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8080
scenario = "calm"
events: list[dict] = []
controls = {"armed": True, "nap_mode": False}


def iso(delta_s: float = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_s)).isoformat()


def add_alert(kind: str, priority: str, title: str, message: str) -> None:
    alert = {"kind": kind, "priority": priority, "at": iso(), "title": title, "message": message}
    events.append(
        {
            "id": len(events) + 1,
            "event": {"type": "alert", "at": iso(), "alert": alert, "snapshot": None, "delivered": True},
            "snapshot_url": None,
            "clip_url": None,
        }
    )


def state() -> dict:
    s = {
        "at": iso(),
        "controls": controls,
        "capabilities": {"vision": True, "push": scenario != "nopush"},
        "camera_up": scenario != "offline",
        "camera_down_for_s": 130.0 if scenario == "offline" else 0.0,
        "sleep": "crying" if scenario == "high" else "asleep",
        "sleep_for_s": 72.0 if scenario == "high" else 5040.0,
        "observation": {"body": "back", "head": "center", "face_covered": False, "in_crib": True,
                        "eyes": "closed", "confidence": 0.91},
        "observation_age_s": 14.0,
        "face_covered_streak": {"pending": 1, "emergency": 2}.get(scenario, 0),
        "stomach_streak": 0,
        "head_left_for_s": 74.0 if scenario == "pending" else 0.0,
    }
    return s


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None: ...

    def send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        global scenario
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/mock":
            scenario = q["scenario"][0]
            if scenario == "high":
                add_alert("crying", "high", "Crying for over a minute", "Crying for 72 s")
            if scenario == "emergency":
                add_alert("face_covered", "emergency", "Face covered", "Nose or mouth may be covered. Seen on 2 checks in a row.")
            return self.send(200, b"{}")
        if scenario == "starting" and url.path.startswith("/api"):
            return self.send(503, b'{"detail":"starting"}')
        if url.path == "/api/state":
            return self.send(200, json.dumps(state()).encode())
        if url.path == "/api/frame.jpg":
            return self.send(503, b"{}") if scenario == "offline" else self.send(200, FRAME, "image/jpeg")
        if url.path == "/api/events":
            since = int(q.get("since_id", ["0"])[0])
            out = [e for e in reversed(events) if e["id"] > since]
            return self.send(200, json.dumps(out).encode())
        if url.path == "/api/timeline":
            now = datetime.now(UTC)
            spans = [("asleep", 22, 14), ("stirring", 14, 13.5), ("asleep", 13.5, 5), ("awake", 5, 4), ("asleep", 4, 0)]
            body = [{"state": s, "start": (now - timedelta(hours=a)).isoformat(), "end": (now - timedelta(hours=b)).isoformat()} for s, a, b in spans]
            return self.send(200, json.dumps(body).encode())
        self.send(404, b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/controls":
            controls.update(body)
            return self.send(202, json.dumps(body).encode())
        if self.path == "/api/reference":
            return self.send(200, json.dumps({"at": iso()}).encode())
        self.send(404, b"{}")


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
