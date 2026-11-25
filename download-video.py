"""Download one MP4 from the GPU container through a Jupyter proxy."""

import argparse
import secrets
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--port", type=int, default=18763)
    args = parser.parse_args()
    video = args.video.resolve()
    if video.suffix.lower() != ".mp4" or not video.is_file():
        parser.error(f"Expected an existing MP4 file: {video}")

    token = secrets.token_urlsafe(24)
    downloaded = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal downloaded
            supplied = parse_qs(urlsplit(self.path).query).get("key", [""])[0]
            if not secrets.compare_digest(supplied.encode("utf-8"), token.encode("ascii")):
                self.send_error(404)
                return

            with video.open("rb") as source:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(video.stat().st_size))
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(video.name, safe='')}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                shutil.copyfileobj(source, self.wfile)
            downloaded = True

        def log_message(self, format, *args):
            pass

    with HTTPServer(("0.0.0.0", args.port), Handler) as server:
        server.timeout = 1
        print(f"Proxy port: {server.server_port}", flush=True)
        print(f"Append to the proxy URL: ?key={token}", flush=True)
        deadline = time.monotonic() + 600
        while not downloaded and time.monotonic() < deadline:
            server.handle_request()
    print("Video sent." if downloaded else "Download link expired.", flush=True)


if __name__ == "__main__":
    main()
