"""Small XML service over the exact offline predictor used for submissions."""
import base64
import binascii
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from uuid import uuid4
import warnings
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from .storage import AssessmentStore


LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SHIPMENT_PATTERN = re.compile(r"[\w.:-]{1,128}", re.UNICODE)
PUBLIC_FIELDS = (
    "assessment_id", "shipment_id", "load_pct", "model_version",
    "created_at", "processing_ms",
)


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status = status
        self.code = code
        self.message = message


def xml_response(root, status_code=200, headers=None):
    return Response(
        ET.tostring(root, encoding="utf-8", xml_declaration=True),
        status_code=status_code, media_type="application/xml", headers=headers,
    )


def error_response(status, code, message, headers=None):
    root = ET.Element("error")
    ET.SubElement(root, "code").text = code
    ET.SubElement(root, "message").text = message
    return xml_response(root, status, headers)


def assessment_element(assessment):
    root = ET.Element("assessment")
    for key in PUBLIC_FIELDS:
        value = assessment[key]
        ET.SubElement(root, key).text = str(value)
    return root


def validate_shipment_id(shipment_id):
    if not SHIPMENT_PATTERN.fullmatch(shipment_id):
        raise ApiError(
            400, "invalid_shipment_id",
            "shipment_id must contain 1–128 letters, digits, underscores, dots, colons or hyphens.",
        )
    return shipment_id


def parse_request(body):
    try:
        root = SafeET.fromstring(
            body, forbid_dtd=True, forbid_entities=True, forbid_external=True,
        )
    except (ET.ParseError, DefusedXmlException, ValueError, LookupError) as exc:
        raise ApiError(400, "invalid_xml", "Malformed XML; DTD and entities are forbidden.") from exc
    if root.tag != "assessment" or root.attrib or (root.text or "").strip():
        raise ApiError(400, "invalid_request", "Expected an assessment element without attributes.")
    children = list(root)
    if len(children) != 2 or sorted(child.tag for child in children) != ["image", "shipment_id"]:
        raise ApiError(400, "invalid_request", "Exactly one shipment_id and one image are required.")
    if any(list(child) or (child.tail or "").strip() for child in children):
        raise ApiError(400, "invalid_request", "Nested elements and extra text are not supported.")
    shipment = root.find("shipment_id")
    image = root.find("image")
    if shipment.attrib or image.attrib != {"encoding": "base64"}:
        raise ApiError(400, "invalid_request", "image must have only encoding=base64; shipment_id has no attributes.")
    shipment_id = validate_shipment_id((shipment.text or "").strip())
    encoded = "".join((image.text or "").split())
    if not encoded:
        raise ApiError(400, "invalid_base64", "Image Base64 must not be empty.")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ApiError(413, "image_too_large", "Decoded image exceeds the 10 MiB limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ApiError(400, "invalid_base64", "Invalid image Base64 encoding.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ApiError(413, "image_too_large", "Decoded image exceeds the 10 MiB limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as decoded:
                if decoded.format not in ("JPEG", "PNG"):
                    raise ApiError(415, "unsupported_image", "Only JPEG and PNG images are supported.")
                suffix = ".jpg" if decoded.format == "JPEG" else ".png"
                if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                    raise ApiError(413, "image_too_large", "Image exceeds the 20 megapixel limit.")
                decoded.verify()
            # verify() checks structure; load() also catches incomplete pixel data.
            with Image.open(BytesIO(raw)) as decoded:
                decoded.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ApiError(413, "image_too_large", "Image exceeds the 20 megapixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ApiError(400, "invalid_image", "Image is damaged or cannot be decoded.") from exc
    return shipment_id, raw, suffix


async def read_limited_body(request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in ("application/xml", "text/xml"):
        raise ApiError(415, "unsupported_media_type", "Use Content-Type: application/xml.")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            parsed_length = int(length)
        except ValueError as exc:
            raise ApiError(400, "invalid_content_length", "Invalid Content-Length.") from exc
        if parsed_length < 0:
            raise ApiError(400, "invalid_content_length", "Invalid Content-Length.")
        if parsed_length > MAX_BODY_BYTES:
            raise ApiError(413, "request_too_large", "XML request exceeds the 16 MiB limit.")
    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise ApiError(413, "request_too_large", "XML request exceeds the 16 MiB limit.")
            body.extend(chunk)
    except ClientDisconnect as exc:
        raise ApiError(400, "incomplete_request", "Client disconnected before completing the XML request.") from exc
    return bytes(body)


def run_assessment(app, body, started):
    shipment_id, raw, suffix = parse_request(body)
    descriptor, image_path = tempfile.mkstemp(prefix="postcode-image-", suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        predictions = app.state.predictor.predict([Path(image_path)])
        if len(predictions) != 1:
            raise ValueError("Predictor returned an unexpected number of predictions")
        load_pct = float(predictions[0])
        if not math.isfinite(load_pct) or not 0 <= load_pct <= 100:
            raise ValueError("Predictor returned a non-finite or out-of-range value")
        assessment = {
            "assessment_id": str(uuid4()),
            "shipment_id": shipment_id,
            "load_pct": load_pct,
            "model_version": app.state.model_version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "processing_ms": round((perf_counter() - started) * 1000, 3),
            "image_sha256": hashlib.sha256(raw).hexdigest(),
        }
        app.state.store.add(assessment)
        return assessment
    finally:
        Path(image_path).unlink(missing_ok=True)


def create_app(
    bundle_dir=Path("runtime/attention518"),
    db_path=Path("runtime/assessments.sqlite3"),
    device="cpu", *, predictor_factory=None, predictor=None,
):
    """Load local weights once at startup; tests can inject an explicit predictor."""
    if predictor is not None and predictor_factory is not None:
        raise ValueError("Provide predictor or predictor_factory, not both")

    @asynccontextmanager
    async def lifespan(app):
        selected = predictor
        if selected is None:
            factory = predictor_factory
            if factory is None:
                from .best_model import Predictor
                factory = Predictor
            selected = await run_in_threadpool(factory, Path(bundle_dir), device=device)
        version = getattr(selected, "model_version", None)
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Predictor must expose a nonempty model_version")
        app.state.predictor = selected
        app.state.model_version = version
        app.state.store = await run_in_threadpool(AssessmentStore, db_path)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(
        title="Postcode XML API", lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.state.ready = False

    @app.get("/", include_in_schema=False)
    async def demo():
        return Response(
            (Path(__file__).parent / "web/index.html").read_text(encoding="utf-8"),
            media_type="text/html",
        )

    @app.exception_handler(ApiError)
    async def api_error_handler(request, exc):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request, exc):
        return error_response(exc.status_code, f"http_{exc.status_code}", str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc):
        return error_response(422, "invalid_parameters", "Invalid request parameters; history limit must be 1–1000.")

    @app.exception_handler(Exception)
    async def internal_error_handler(request, exc):
        LOGGER.error("Request processing failed", exc_info=(type(exc), exc, exc.__traceback__))
        return error_response(500, "internal_error", "Processing failed; consult the server log.")

    @app.get("/health")
    async def health():
        root = ET.Element("health")
        ET.SubElement(root, "status").text = "ready" if app.state.ready else "unavailable"
        if app.state.ready:
            ET.SubElement(root, "model_version").text = app.state.model_version
        return xml_response(root, 200 if app.state.ready else 503)

    @app.post("/assessments")
    async def assess(request: Request):
        started = perf_counter()
        body = await read_limited_body(request)
        assessment = await run_in_threadpool(run_assessment, app, body, started)
        return xml_response(assessment_element(assessment), 201)

    @app.get("/shipments/{shipment_id}")
    async def latest(shipment_id: str):
        validate_shipment_id(shipment_id)
        assessment = await run_in_threadpool(app.state.store.latest, shipment_id)
        if assessment is None:
            raise ApiError(404, "shipment_not_found", "No assessment exists for this shipment.")
        return xml_response(assessment_element(assessment))

    @app.get("/shipments/{shipment_id}/history")
    async def history(shipment_id: str, limit: int = Query(default=100, ge=1, le=1000)):
        validate_shipment_id(shipment_id)
        records = await run_in_threadpool(app.state.store.history, shipment_id, limit)
        if not records:
            raise ApiError(404, "shipment_not_found", "No assessment exists for this shipment.")
        root = ET.Element("assessments", {"shipment_id": shipment_id, "count": str(len(records))})
        for record in records:
            root.append(assessment_element(record))
        return xml_response(root)

    return app
