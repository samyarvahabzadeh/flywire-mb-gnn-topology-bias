"""Local synthetic archive-scope tests; no uploads, credentials or registration."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

from mbtopology.runtime import Fail, sha256

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_osf_review", ROOT / "scripts/prepare_osf_review.py"
)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def test_review_archive_whitelist_excludes_private_and_dataset_payloads(tmp_path):
    (tmp_path / "README.md").write_text("SYNTHETIC PUBLIC REVIEW FIXTURE")
    (tmp_path / ".gitattributes").write_text("configs/** text eol=lf\n")
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/MANIFEST.json").write_text('{"scope":"SYNTHETIC_MANIFEST_ONLY"}')
    (tmp_path / "data/raw/fixture.csv.gz").write_bytes(b"SYNTHETIC_RAW_PAYLOAD_SENTINEL")
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local/token.txt").write_text("SYNTHETIC_PRIVATE_SENTINEL")
    selected = PACKAGE.safe_artifacts(
        tmp_path, ["README.md", ".gitattributes", "data/MANIFEST.json"]
    )
    destination = tmp_path / "review.zip"
    result = PACKAGE.write_archive(
        tmp_path,
        destination,
        selected,
        {"REVIEW_NOTES.md": "SYNTHETIC_REVIEW_ONLY"},
        {"scope": "SYNTHETIC_UNIT_FIXTURE"},
    )
    assert result["archive_sha256"] == sha256(destination)
    assert result["uploaded"] is False and result["registered"] is False
    with zipfile.ZipFile(destination) as archive:
        contents = b"\n".join(archive.read(name) for name in archive.namelist())
        assert "project/.gitattributes" in archive.namelist()
        assert b"SYNTHETIC_PRIVATE_SENTINEL" not in contents
        assert b"SYNTHETIC_RAW_PAYLOAD_SENTINEL" not in contents
    with pytest.raises(Fail, match="hidden/private"):
        PACKAGE.safe_artifacts(tmp_path, [".local/token.txt"])
    with pytest.raises(Fail, match="dataset payloads"):
        PACKAGE.safe_artifacts(tmp_path, ["data/raw/fixture.csv.gz"])


def test_review_artifact_cannot_escape_root_or_include_live_evidence(tmp_path):
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    (tmp_path / "outside.txt").write_text("SYNTHETIC_OUTSIDE_SENTINEL")
    (root / "configs/osf_registration.json").write_text('{"scope":"SYNTHETIC"}')
    with pytest.raises(Fail, match="outside"):
        PACKAGE.safe_artifacts(root, ["../outside.txt"])
    with pytest.raises(Fail, match="live registration"):
        PACKAGE.safe_artifacts(root, ["configs/osf_registration.json"])
