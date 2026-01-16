"""HTTP API server for Cloud Run deployment.

Simple async HTTP server that wraps the mycelium solver.
"""

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Port from environment (Cloud Run sets PORT=8080)
PORT = int(os.getenv("PORT", 8080))


class MyceliumHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mycelium API."""

    solver = None  # Lazy initialized

    @classmethod
    def get_solver(cls):
        """Get or create the solver instance."""
        if cls.solver is None:
            from mycelium.solver import Solver
            cls.solver = Solver()
            logger.info("[server] Solver initialized")
        return cls.solver

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"status": "ok", "service": "mycelium"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            from mycelium.step_signatures import StepSignatureDB
            step_db = StepSignatureDB()
            response = {
                "status": "ok",
                "signatures": step_db.count_signatures(),
                "provider": os.getenv("MYCELIUM_PROVIDER", "local"),
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (solve problem)."""
        if self.path == "/solve":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()

            try:
                data = json.loads(body)
                problem = data.get("problem")
                if not problem:
                    self.send_error(400, "Missing 'problem' field")
                    return

                # Run solver asynchronously
                solver = self.get_solver()
                result = asyncio.run(solver.solve(problem=problem))

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                response = {
                    "answer": result.answer,
                    "steps": result.total_steps,
                    "signatures_matched": result.signatures_matched,
                    "dsl_injections": result.steps_with_injection,
                    "elapsed_ms": result.elapsed_ms,
                }
                self.wfile.write(json.dumps(response).encode())

            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                logger.exception("[server] Error solving problem")
                self.send_error(500, str(e))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging, use our logger instead."""
        logger.info("[server] %s", format % args)


def run_server(port: int = PORT):
    """Run the HTTP server."""
    logging.basicConfig(level=logging.INFO)
    server = HTTPServer(("0.0.0.0", port), MyceliumHandler)
    logger.info(f"[server] Starting on port {port}")
    logger.info(f"[server] Provider: {os.getenv('MYCELIUM_PROVIDER', 'local')}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
