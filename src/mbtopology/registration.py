from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .runtime import Fail, require

REPOSITORY = "samyarvahabzadeh/flywire-mb-gnn-topology-bias"
TAG = "v1.0-prereg"
TAG_REF = f"refs/tags/{TAG}"
TAG_OBJECT = "92b818d2ccd9c3c2c5f4a3bc74408d279659f05e"
COMMIT = "5e2c0b384252c1415f4e0fd61fd973aa28fcdb0a"
CHECKOUT_SHA256 = "3fc010a514841e6c1aef9e4f5dc016f94c974197fbdc91e3d307534f5e727be6"
BLOB_SHA256 = "9ec100c10124c483855513ff119d154a70dd712bfd6fa7c0a0f69d0e61942d82"
DOCUMENT = "PREREGISTRATION.md"


def git_bytes(root, *args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Fail(f"FAIL: GitHub tag gate: local {TAG} is missing or unreadable") from exc


@lru_cache(maxsize=1)
def verify_public_tag():
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/git/ref/tags/{TAG}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "mbtopology"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError) as exc:
        raise Fail(f"FAIL: GitHub tag gate: could not verify public {TAG}: {exc}") from exc
    obj = payload.get("object", {})
    require(
        payload.get("ref") == TAG_REF and obj.get("type") == "tag" and obj.get("sha") == TAG_OBJECT,
        "GitHub tag gate: public annotated tag does not match the frozen tag object",
    )


def require_registered(root):
    root = Path(root)
    document = root / DOCUMENT
    require(document.is_file(), "GitHub tag gate: PREREGISTRATION.md is missing")
    checkout = document.read_bytes()
    checkout_hash = hashlib.sha256(checkout).hexdigest()
    require(
        checkout_hash in {CHECKOUT_SHA256, BLOB_SHA256},
        "GitHub tag gate: working preregistration hash does not match",
    )
    require(
        git_bytes(root, "rev-parse", TAG_REF).decode().strip() == TAG_OBJECT
        and git_bytes(root, "rev-parse", f"{TAG_REF}^{{commit}}").decode().strip() == COMMIT,
        "GitHub tag gate: local tag or its target does not match",
    )
    blob = git_bytes(root, "show", f"{TAG_REF}:{DOCUMENT}")
    require(
        hashlib.sha256(blob).hexdigest() == BLOB_SHA256
        and hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest() == CHECKOUT_SHA256
        and checkout in (blob, blob.replace(b"\n", b"\r\n")),
        "GitHub tag gate: tagged document and checkout do not match",
    )
    verify_public_tag()
    return {
        "type": "public_github_annotated_tag",
        "tag": TAG,
        "tag_url": f"https://github.com/{REPOSITORY}/tree/{TAG}",
        "tag_object": TAG_OBJECT,
        "tagged_commit": COMMIT,
        "tagged_at": git_bytes(root, "for-each-ref", TAG_REF, "--format=%(taggerdate:iso-strict)")
        .decode()
        .strip(),
        "preregistration_sha256": CHECKOUT_SHA256,
        "tagged_blob_sha256": BLOB_SHA256,
        "checkout_sha256": checkout_hash,
    }
