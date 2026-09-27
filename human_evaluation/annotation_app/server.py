"""Local-only HTTP server for the annotation website."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .storage import AnnotationStore, load_samples, validate_annotator_id
from .translations import attach_translations, load_translations


PREFERRED_DATASET_ORDER = ("msmarco", "text2sql", "theoremqa-theorems")
MAX_REQUEST_BYTES = 64 * 1024


def order_samples(
    samples: list[dict[str, str]], annotator_id: str
) -> tuple[list[str], list[dict[str, str]]]:
    """Group datasets and deterministically shuffle each annotator's block."""
    present = {sample["dataset"] for sample in samples}
    dataset_order = [name for name in PREFERRED_DATASET_ORDER if name in present]
    dataset_order.extend(sorted(present - set(dataset_order)))
    ordered: list[dict[str, str]] = []
    for dataset in dataset_order:
        block = [sample for sample in samples if sample["dataset"] == dataset]
        block.sort(
            key=lambda sample: hashlib.sha256(
                f"{annotator_id}\0{dataset}\0{sample['sample_id']}".encode("utf-8")
            ).digest()
        )
        ordered.extend(block)
    return dataset_order, ordered


def build_server(
    root: str | Path,
    annotator_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Build a configured server without starting its event loop."""
    application_root = Path(root).resolve()
    safe_annotator_id = validate_annotator_id(annotator_id)
    samples = load_samples(application_root / "data" / "samples.json")
    translations = load_translations(
        application_root / "data" / "translations.zh-CN.json", samples
    )
    display_samples = attach_translations(samples, translations)
    dataset_order, ordered_samples = order_samples(display_samples, safe_annotator_id)
    store = AnnotationStore(
        samples,
        application_root / "annotations" / f"{safe_annotator_id}.csv",
        safe_annotator_id,
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "HumanAnnotation/1.0"

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path == "/api/session":
                self._send_json(
                    200,
                    {
                        "annotator_id": safe_annotator_id,
                        "dataset_order": dataset_order,
                        "samples": ordered_samples,
                        "scores": store.scores(),
                    },
                )
                return
            file_path = self._public_file(path)
            if file_path is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_file(file_path)

        def do_POST(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path != "/api/annotations":
                self._send_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_json()
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                sample_id = payload.get("sample_id")
                score = payload.get("score")
                if not isinstance(sample_id, str):
                    raise ValueError("sample_id must be a string")
                store.save(sample_id, score)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            scores = store.scores()
            self._send_json(
                200,
                {
                    "sample_id": sample_id,
                    "saved_score": score,
                    "completed": len(scores),
                    "total": len(samples),
                },
            )

        def _read_json(self) -> Any:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _public_file(self, request_path: str) -> Path | None:
            exact = {
                "/": application_root / "web" / "index.html",
                "/index.html": application_root / "web" / "index.html",
                "/app.js": application_root / "web" / "app.js",
                "/styles.css": application_root / "web" / "styles.css",
                "/favicon.svg": application_root / "web" / "favicon.svg",
            }
            if request_path in exact:
                candidate = exact[request_path]
                allowed_root = application_root / "web"
            elif request_path.startswith("/vendor/"):
                relative_text = request_path.removeprefix("/vendor/")
                relative = Path(relative_text)
                if (
                    not relative_text
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or "." in relative.parts
                ):
                    return None
                allowed_root = application_root / "web" / "vendor"
                candidate = allowed_root / relative
            elif request_path.startswith("/guidelines/"):
                filename = request_path.removeprefix("/guidelines/")
                allowed = {f"{dataset}.md" for dataset in dataset_order}
                if filename not in allowed:
                    return None
                allowed_root = application_root / "guidelines"
                candidate = allowed_root / filename
            else:
                return None
            resolved_root = allowed_root.resolve()
            resolved = candidate.resolve()
            if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
                return None
            return resolved

        def _send_file(self, path: Path) -> None:
            content = path.read_bytes()
            content_type, _ = mimetypes.guess_type(path.name)
            if path.suffix == ".md":
                content_type = "text/markdown"
            self.send_response(200)
            self._common_headers()
            self.send_header(
                "Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8"
            )
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, status: int, payload: Any) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _common_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "font-src 'self'; img-src 'self' data:; connect-src 'self'",
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
