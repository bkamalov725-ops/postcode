"""XML contract, safe image handling and durable history without GPU weights."""
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from postcode_ml.service import ApiError, create_app, read_limited_body


def image_bytes(format="PNG"):
    stream = BytesIO()
    Image.new("RGB", (16, 12), (20, 100, 40)).save(stream, format=format)
    return stream.getvalue()


def request_xml(shipment_id="00123", raw=None):
    encoded = base64.b64encode(image_bytes() if raw is None else raw).decode("ascii")
    return f'<assessment><shipment_id>{shipment_id}</shipment_id><image encoding="base64">{encoded}</image></assessment>'


class FakePredictor:
    model_version = "test-bundle-v1"

    def __init__(self):
        self.result = 73.25
        self.seen = []

    def predict(self, paths):
        self.seen.extend(paths)
        # Actual temporary file is readable for the lifetime of inference.
        for path in paths:
            with Image.open(path) as image:
                image.load()
        return [self.result] * len(paths)


class XmlApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "history.sqlite3"
        self.predictor = FakePredictor()
        self.app = create_app(db_path=self.db, predictor=self.predictor)
        self.client_context = TestClient(self.app, raise_server_exceptions=False)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp.cleanup()

    def post(self, body=None, **kwargs):
        return self.client.post(
            "/assessments", content=request_xml() if body is None else body,
            headers={"Content-Type": "application/xml"}, **kwargs,
        )

    def assert_error(self, response, status, code=None):
        self.assertEqual(response.status_code, status, response.text)
        self.assertIn("application/xml", response.headers["content-type"])
        root = ET.fromstring(response.content)
        self.assertEqual(root.tag, "error")
        if code is not None:
            self.assertEqual(root.findtext("code"), code)
        self.assertTrue(root.findtext("message"))

    def test_create_preserves_identifier_and_returns_metadata(self):
        response = self.post()
        self.assertEqual(response.status_code, 201, response.text)
        result = ET.fromstring(response.content)
        self.assertEqual(result.findtext("shipment_id"), "00123")
        self.assertEqual(float(result.findtext("load_pct")), 73.25)
        self.assertEqual(result.findtext("model_version"), "test-bundle-v1")
        self.assertTrue(result.findtext("created_at").endswith("Z"))
        self.assertGreaterEqual(float(result.findtext("processing_ms")), 0)
        self.assertTrue(result.findtext("assessment_id"))
        self.assertTrue(self.db.exists())
        self.assertTrue(all(not path.exists() for path in self.predictor.seen))

    def test_jpeg_and_wrapped_base64(self):
        xml = request_xml(raw=image_bytes("JPEG"))
        xml = xml.replace('<image encoding="base64">', '<image encoding="base64">\n  ')
        self.assertEqual(self.post(xml).status_code, 201)

    def test_repeated_post_creates_history_and_get_does_not_infer(self):
        first = ET.fromstring(self.post().content)
        self.predictor.result = 42.5
        second = ET.fromstring(self.post().content)
        self.assertNotEqual(first.findtext("assessment_id"), second.findtext("assessment_id"))
        latest = ET.fromstring(self.client.get("/shipments/00123").content)
        self.assertEqual(latest.findtext("assessment_id"), second.findtext("assessment_id"))
        history = ET.fromstring(self.client.get("/shipments/00123/history").content)
        self.assertEqual(history.attrib["count"], "2")
        self.assertEqual([x.findtext("load_pct") for x in history], ["42.5", "73.25"])
        limited = ET.fromstring(self.client.get("/shipments/00123/history?limit=1").content)
        self.assertEqual(len(limited), 1)
        self.assertEqual(len(self.predictor.seen), 2)

    def test_persistence_survives_new_application(self):
        stored = self.post().content
        new_predictor = FakePredictor()
        with TestClient(create_app(db_path=self.db, predictor=new_predictor)) as other:
            self.assertEqual(other.get("/shipments/00123").content, stored)
        self.assertEqual(new_predictor.seen, [])

    def test_model_loaded_once_through_factory(self):
        calls = []

        def factory(bundle_dir, device):
            calls.append((bundle_dir, device))
            return FakePredictor()

        app = create_app("local-weights", self.db, "cpu", predictor_factory=factory)
        with TestClient(app) as client:
            for _ in range(2):
                self.assertEqual(client.post("/assessments", content=request_xml(), headers={"Content-Type": "application/xml"}).status_code, 201)
        self.assertEqual(calls, [(Path("local-weights"), "cpu")])

    def test_missing_weights_fail_startup(self):
        def failing_factory(*args, **kwargs):
            raise FileNotFoundError("Missing offline weights")

        with self.assertRaises(FileNotFoundError):
            with TestClient(create_app(db_path=self.db, predictor_factory=failing_factory)):
                self.fail("Service must not become ready without weights")

    def test_health_responds_while_predictor_is_busy(self):
        entered, release = threading.Event(), threading.Event()
        original = self.predictor.predict

        def slow_predict(paths):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test inference timed out")
            return original(paths)

        self.predictor.predict = slow_predict
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.post)
            try:
                self.assertTrue(entered.wait(5))
                health = self.client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ET.fromstring(health.content).findtext("status"), "ready")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).status_code, 201)

    def test_missing_shipment_returns_xml_404(self):
        for endpoint in ("/shipments/unknown", "/shipments/unknown/history", "/absent"):
            with self.subTest(endpoint=endpoint):
                self.assert_error(self.client.get(endpoint), 404)

    def test_method_and_parameter_errors_use_xml(self):
        self.assert_error(self.client.get("/assessments"), 405)
        for limit in ("0", "1001", "no"):
            self.assert_error(self.client.get(f"/shipments/00123/history?limit={limit}"), 422)

    def test_non_xml_body_rejected(self):
        self.assert_error(self.client.post("/assessments", json={"shipment_id": "1"}), 415, "unsupported_media_type")

    def test_malformed_xml_and_strict_schema(self):
        for body in (
            "<assessment>", "", "<request/>",
            request_xml().replace("</assessment>", "<extra/></assessment>"),
            request_xml().replace("</shipment_id>", "</shipment_id><shipment_id>2</shipment_id>"),
            request_xml().replace("<assessment>", '<assessment unexpected="yes">'),
            request_xml().replace("00123", "<number>00123</number>"),
            request_xml().replace('encoding="base64"', 'encoding="url"'),
            request_xml().replace("00123", ""),
            request_xml().replace("00123", "a/b"),
        ):
            with self.subTest(body=body[:100]):
                self.assert_error(self.post(body), 400)
        self.assertEqual(self.predictor.seen, [])

    def test_dtd_and_entities_rejected_in_utf8_and_utf16(self):
        payload = '<!DOCTYPE assessment [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>' + request_xml("&xxe;")
        for encoding in ("utf-8", "utf-16"):
            body = f'<?xml version="1.0" encoding="{encoding}"?>{payload}'.encode(encoding)
            self.assert_error(self.post(body), 400, "invalid_xml")
        self.assert_error(self.post("<!DOCTYPE assessment>" + request_xml()), 400, "invalid_xml")
        self.assertEqual(self.predictor.seen, [])

    def test_invalid_base64_and_corrupt_image(self):
        for encoded in ("%%%", "", "a", "ж", "abcd===junk"):
            body = f'<assessment><shipment_id>1</shipment_id><image encoding="base64">{encoded}</image></assessment>'
            self.assert_error(self.post(body), 400)
        self.assert_error(self.post(request_xml(raw=b"not a picture")), 400, "invalid_image")
        self.assert_error(self.post(request_xml(raw=image_bytes("PNG")[:40])), 400, "invalid_image")
        self.assertEqual(self.predictor.seen, [])

    def test_unsupported_actual_image_format(self):
        self.assert_error(self.post(request_xml(raw=image_bytes("GIF"))), 415, "unsupported_image")

    def test_body_limits_include_streamed_request(self):
        with patch("postcode_ml.service.MAX_BODY_BYTES", 100):
            self.assert_error(self.post(), 413, "request_too_large")
            # Generator omits Content-Length; streaming limit must still apply.
            response = self.post(iter([b" " * 60, b" " * 60]))
            self.assert_error(response, 413, "request_too_large")
        self.assertEqual(self.predictor.seen, [])

    def test_decoded_size_and_pixel_limits(self):
        with patch("postcode_ml.service.MAX_IMAGE_BYTES", 10):
            self.assert_error(self.post(), 413, "image_too_large")
        with patch("postcode_ml.service.MAX_IMAGE_PIXELS", 100):
            self.assert_error(self.post(), 413, "image_too_large")
        self.assertEqual(self.predictor.seen, [])

    def test_interrupted_request_is_validation_error(self):
        messages = iter([
            {"type": "http.request", "body": b"<assessment>", "more_body": True},
            {"type": "http.disconnect"},
        ])

        async def receive():
            return next(messages)

        request = Request({"type": "http", "headers": [(b"content-type", b"application/xml")]}, receive)
        with self.assertRaises(ApiError) as captured:
            asyncio.run(read_limited_body(request))
        self.assertEqual(captured.exception.status, 400)
        self.assertEqual(captured.exception.code, "incomplete_request")

    def test_invalid_predictions_return_generic_error_and_remove_image(self):
        for prediction in (float("nan"), float("inf"), -1, 101):
            self.predictor.result = prediction
            with self.assertLogs("postcode_ml.service", level="ERROR"):
                self.assert_error(self.post(), 500, "internal_error")
        self.assertTrue(all(not path.exists() for path in self.predictor.seen))
        self.assert_error(self.client.get("/shipments/00123"), 404)

    def test_inference_failure_does_not_expose_internal_paths(self):
        def failed_predict(paths):
            self.predictor.seen.extend(paths)
            raise RuntimeError("Secret internal path and traceback")

        self.predictor.predict = failed_predict
        with self.assertLogs("postcode_ml.service", level="ERROR"):
            response = self.post()
        self.assert_error(response, 500, "internal_error")
        self.assertNotIn("Secret", response.text)
        self.assertTrue(all(not path.exists() for path in self.predictor.seen))


if __name__ == "__main__":
    unittest.main()
