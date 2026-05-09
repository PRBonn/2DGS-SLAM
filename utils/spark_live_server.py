import errno
import os
import shutil
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def _guess_lan_ipv4():
    """Best-effort local IPv4 for sharing the viewer on Wi-Fi / LAN."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _spark_index_source_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "gui", "spark_viewer", "index.html"
    )


class _SparkLiveHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        path_only = urlparse(self.path).path
        base = os.path.basename(path_only)
        if base in ("live.ply", "traj_live.json"):
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt, *args):
        return


def start_spark_live_server(
    workspace_dir: str, port: int = 8765, max_port_tries: int = 32
):
    """
    Serves workspace_dir (contains index.html + live.ply) on 0.0.0.0:port.
    If *port* is already in use (e.g. another SLAM or viewer), tries the next
    ports up to *max_port_tries* times.
    Returns (shutdown_callable, bound_port).
    """
    os.makedirs(workspace_dir, exist_ok=True)
    src = _spark_index_source_path()
    dst = os.path.join(workspace_dir, "index.html")
    shutil.copyfile(src, dst)

    handler = lambda *args, **kwargs: _SparkLiveHTTPRequestHandler(
        *args, directory=workspace_dir, **kwargs
    )
    server = None
    bound_port = None
    last_err = None
    for candidate in range(port, port + max_port_tries):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", candidate), handler)
            bound_port = candidate
            break
        except OSError as e:
            last_err = e
            if e.errno != errno.EADDRINUSE:
                raise
            continue
    if server is None or bound_port is None:
        raise OSError(
            f"Spark live: could not bind to ports {port}..{port + max_port_tries - 1} ({last_err})"
        ) from last_err

    if bound_port != port:
        print(
            f"[SparkLive] Port {port} was busy; serving on {bound_port} instead."
        )

    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[SparkLive] This machine: http://127.0.0.1:{bound_port}/")
    lan = _guess_lan_ipv4()
    if lan:
        print(f"[SparkLive] Other devices (same network): http://{lan}:{bound_port}/")

    def shutdown():
        server.shutdown()
        server.server_close()

    return shutdown, bound_port
