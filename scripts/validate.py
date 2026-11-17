"""Run local engineering checks and preserve their actual logs on project storage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mbtopology.runtime import atomic_json, now, provenance


def main():
    root = Path(__file__).resolve().parents[1]
    temporary = (root / ".tmp" / "validation-tests").resolve()
    temporary.relative_to(root)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "TEMP": str(temporary.parent),
        "TMP": str(temporary.parent),
        "TMPDIR": str(temporary.parent),
    }
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--basetemp",
            str(temporary),
            "--junitxml",
            str(root / "reports" / "engineering_tests.xml"),
        ],
    ]
    outcomes = []
    for index, command in enumerate(commands):
        result = subprocess.run(
            command,
            check=False,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log = log_dir / f"validation_{index}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        outcomes.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "log": log.relative_to(root).as_posix(),
                "completed_at": now(),
            }
        )
        print(result.stdout + result.stderr, end="")
        if result.returncode:
            break
    report = {
        "status": "PASS" if all(o["exit_code"] == 0 for o in outcomes) else "FAIL",
        "scope": "SYNTHETIC_ENGINEERING_TESTS_ONLY",
        "checks": outcomes,
        "provenance": provenance(root, {}),
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports" / "validation.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
