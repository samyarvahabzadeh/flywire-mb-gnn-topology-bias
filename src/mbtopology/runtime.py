from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml


class Fail(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Fail("FAIL: " + message)


def now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_file(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid4().hex + ".partial" + path.suffix)
    writer(temporary)
    os.replace(temporary, path)


def disk_guard(root, reserve, needed=0):
    require(
        shutil.disk_usage(root).free >= reserve + needed,
        f"insufficient free disk space; need {reserve + needed} bytes",
    )


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require(
        config["dataset"] == "fafb" and str(config["snapshot"]) == "783",
        "this stage is frozen to dataset=fafb, snapshot=783",
    )
    require(0 < config["checkpoint_seconds"] <= 900, "checkpoint interval must be <=900s")
    require(config["graph"]["threshold"] == 5, "primary threshold must equal 5")
    return config


def provenance(root, config):
    root = Path(root)
    files = sorted(
        p
        for base in ("src", "configs")
        for p in (root / base).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not any(part.endswith(".egg-info") for part in p.parts)
    )
    sources = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for name in ("polars", "pyarrow", "scipy", "numpy", "torch", "torch-geometric"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    return {
        "git_commit": commit,
        "config_sha256": fingerprint(config),
        "source_sha256": fingerprint(sources),
        "source_files": sources,
        "packages": packages,
    }


class Run:
    def __init__(self, root, config, stage):
        self.root, self.config, self.stage = Path(root), config, stage
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.path = self.root / "checkpoints" / f"{stage}.json"
        self.state = {
            "stage": stage,
            "status": "running",
            "started_at": now(),
            "provenance": provenance(root, config),
            "progress": {},
        }
        self.error = None

    def save(self, **progress):
        with self.lock:
            if self.error:
                raise Fail("FAIL: periodic checkpoint write failed") from self.error
            self.state["progress"].update(progress)
            self.state["checkpoint_at"] = now()
            atomic_json(self.path, self.state)

    def heartbeat(self):
        while not self.stop.wait(self.config["checkpoint_seconds"]):
            try:
                self.save()
            except (OSError, TypeError, ValueError, Fail) as exc:
                self.error = exc
                return

    def __enter__(self):
        self.save()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        self.state["status"] = "failed" if kind else "complete"
        if kind:
            self.state["error_type"] = kind.__name__
        self.save()
        log = self.root / "logs" / f"{self.stage}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.state, ensure_ascii=False) + "\n")


class Exclusive:
    def __init__(self, root):
        self.path = Path(root) / "checkpoints" / "pipeline.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_at": now()}, stream)
        except FileExistsError:
            raise Fail("FAIL: pipeline.lock exists; verify no other writer before removing it")

    def __exit__(self, *args):
        self.path.unlink()
