#!/usr/bin/env python3
"""Start the local human-annotation website."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path
from typing import Sequence

from annotation_app.server import build_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local, auto-saving human annotation website."
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help="Your assigned ID, for example annotator_1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local HTTP port (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser automatically",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        server = build_server(root, args.annotator, port=args.port)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot start annotation website: {exc}") from exc
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    output = root / "annotations" / f"{args.annotator}.csv"
    print(f"Annotation website: {url}")
    print(f"Scores are saved after every click to: {output}")
    print("Press Ctrl+C to stop. Your saved progress will remain on disk.")
    if not args.no_browser:
        opener = threading.Timer(0.25, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
