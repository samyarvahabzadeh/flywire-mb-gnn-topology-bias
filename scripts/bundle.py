"""Package only explicitly allowed project artifacts for persistent molab storage."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / name
        for name in (
            "README.md",
            "PREREGISTRATION.md",
            "EXPERIMENT_SPEC.md",
            "DECISIONS.md",
            "RUN_WEEK1.md",
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
            "data/MANIFEST.json",
        )
    ]
    for directory, pattern in (
        ("src", "*.py"),
        ("tests", "*.py"),
        ("configs", "*.yaml"),
        ("scripts", "*.py"),
        ("scripts", "*.ps1"),
        ("reports", "*.json"),
        ("reports", "*.md"),
        ("reports", "*.xml"),
    ):
        files.extend((root / directory).rglob(pattern))
    selected = sorted({p for p in files if p.is_file() and "__pycache__" not in p.parts})
    token_path = root / ".local" / "codex_token.txt"
    secret = token_path.read_bytes().strip() if token_path.exists() else b""
    output = root / "dist" / "week1-source.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in selected:
            content = path.read_bytes()
            if secret and secret in content:
                raise RuntimeError("credential unexpectedly present in a selected artifact")
            bundle.writestr(path.relative_to(root).as_posix(), content)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": checksum,
        "files": [p.relative_to(root).as_posix() for p in selected],
        "contains_raw_data": False,
        "contains_credentials": False,
    }
    (output.parent / "week1-source.manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
