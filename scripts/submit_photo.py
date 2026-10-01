"""Send one local JPEG/PNG and a shipment ID to the XML API (stdlib only)."""
import argparse
import base64
from pathlib import Path
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--shipment-id", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = ET.Element("assessment")
    ET.SubElement(root, "shipment_id").text = args.shipment_id
    ET.SubElement(root, "image", encoding="base64").text = base64.b64encode(
        args.image.read_bytes()).decode("ascii")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/assessments",
        data=ET.tostring(root, encoding="utf-8", xml_declaration=True),
        headers={"Content-Type": "application/xml", "Accept": "application/xml"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", errors="replace"))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
