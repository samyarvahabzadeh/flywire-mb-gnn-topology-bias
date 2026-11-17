"""Inspect status/redirect metadata for the requested resource without saving its payload."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import requests

from mbtopology.ingest import BASE, clean_url, read_token
from mbtopology.runtime import atomic_json, now


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def main():
    root = Path(__file__).resolve().parents[1]
    token = read_token(root / ".local" / "codex_token.txt")
    report = {"created_at": now(), "product": "connections", "dataset": "fafb", "snapshot": "783"}
    try:
        with requests.get(
            BASE + "/api/download_resource",
            params={
                "dataset": "fafb",
                "data_version": "783",
                "data_product": "connections",
                "api_token": token,
            },
            stream=True,
            timeout=(30, 60),
        ) as response:
            report["http_status"] = response.status_code
            report["content_type"] = response.headers.get("Content-Type")
            report["route"] = [
                {"status": r.status_code, "url": clean_url(r.url)}
                for r in [*response.history, response]
            ]
            if response.status_code >= 400:
                body = next(response.iter_content(4096), b"").decode("utf-8", errors="replace")
                body = body.replace(token, "[REDACTED]")
                body = re.sub(r"https?://[^\s<>\"']+", "[URL REDACTED]", body)
                parser = PlainText()
                parser.feed(body)
                report["error_text"] = " ".join(" ".join(parser.parts).split())[:600]
    except requests.RequestException as exc:
        report["error_type"] = type(exc).__name__
    try:
        with requests.get(
            BASE + "/api/download",
            params={"dataset": "fafb", "data_version": "783", "api_token": token},
            timeout=(30, 60),
        ) as response:
            report["catalogue_response"] = {
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
            }
            if "application/json" in response.headers.get("Content-Type", ""):
                document = response.json()
                report["catalogue_response"]["json_type"] = type(document).__name__
                report["catalogue_response"]["top_level_keys"] = (
                    list(document) if isinstance(document, dict) else None
                )
                sample = document if isinstance(document, dict) else document[:1]
                safe_sample = json.loads(json.dumps(sample).replace(token, "[REDACTED]"))
                report["catalogue_response"]["sample"] = safe_sample
    except requests.RequestException as exc:
        report["catalogue_response"] = {"error_type": type(exc).__name__}
    atomic_json(root / "reports" / "connection_endpoint_probe.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
