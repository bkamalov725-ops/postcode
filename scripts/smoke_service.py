"""Exercise the real local HTTP server and offline model, then stop the server."""
import argparse
import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "runtime/model_bundle")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/http_smoke.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    database = args.output.parent.resolve() / f"http_smoke_{token}.sqlite3"
    log_path = args.output.parent / f"http_smoke_{token}.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    command = [sys.executable, str(ROOT / "serve.py"), "--bundle-dir", str(args.bundle.resolve()),
               "--db-path", str(database), "--host", "127.0.0.1", "--port", str(port), "--threads", "4"]
    shipment_id = "000123"

    def request(path, body=None):
        req = urllib.request.Request(url + path, data=body,
                                     headers={"Content-Type": "application/xml"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, response.headers.get_content_type(), ET.fromstring(response.read())
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get_content_type(), ET.fromstring(error.read())

    def start_server(log):
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        deadline = time.monotonic() + 90
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: see {log_path}")
                try:
                    status, media, root = request("/health")
                    if status == 200 and root.findtext("status") == "ready":
                        assert media == "application/xml"
                        return process
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.25)
            raise TimeoutError(f"Server did not become ready: see {log_path}")
        except BaseException:
            stop_server(process)
            raise

    def stop_server(process):
        if process.poll() is not None:
            return
        if sys.platform == "win32":
            # The Windows venv redirector can spawn another python.exe. Stop
            # only this launched process tree, not unrelated Python processes.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    with log_path.open("w", encoding="utf-8") as log:
        process = start_server(log)
        try:
            root = ET.Element("assessment")
            ET.SubElement(root, "shipment_id").text = shipment_id
            ET.SubElement(root, "image", encoding="base64").text = base64.b64encode(args.image.read_bytes()).decode("ascii")
            body = ET.tostring(root, encoding="utf-8")
            results = []
            for _ in range(2):
                started = time.perf_counter()
                status, media, root = request("/assessments", body)
                assert status == 201 and media == "application/xml", ET.tostring(root)
                assert root.findtext("shipment_id") == shipment_id
                results.append({"wall_seconds": time.perf_counter() - started,
                                **{child.tag: child.text for child in root}})
            assert results[0]["assessment_id"] != results[1]["assessment_id"]
            status, _, latest = request("/shipments/" + shipment_id)
            assert status == 200 and latest.findtext("assessment_id") == results[-1]["assessment_id"]
            status, _, history = request("/shipments/" + shipment_id + "/history")
            assert status == 200 and len(history) == 2
            assert request("/assessments", b"<broken>")[0] == 400
            assert request("/shipments/does-not-exist")[0] == 404
        finally:
            stop_server(process)
        # Start a different OS process to check persistence across a real restart.
        process = start_server(log)
        try:
            status, _, latest = request("/shipments/" + shipment_id)
            assert status == 200 and latest.findtext("assessment_id") == results[-1]["assessment_id"]
        finally:
            stop_server(process)
    report = {"status": "passed", "requests": results, "history_count": 2,
              "persisted_after_process_restart": True, "xml_error_statuses": [400, 404],
              "database": str(database), "log": str(log_path)}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
