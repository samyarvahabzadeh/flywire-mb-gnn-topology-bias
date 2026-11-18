"""Read-back verified, immutable sweep shards on an external filesystem."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .runtime import atomic_file, require


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(document):
    return json.dumps(document, sort_keys=True, indent=2).encode() + b"\n"


class ArchiveStore:
    """Publish a receipt only after verifying the bytes stored outside Molab."""

    def __init__(self, fs, prefix, plan_hash):
        self.fs = fs
        self.prefix = prefix.rstrip("/")
        self.plan_hash = plan_hash
        self.fs.makedirs(self.prefix, exist_ok=True)

    def path(self, name):
        require(PurePosixPath(name).name == name and name not in (".", ".."), "invalid object name")
        return f"{self.prefix}/{name}"

    def list_names(self):
        self.fs.invalidate_cache(self.prefix)
        return {PurePosixPath(name).name for name in self.fs.ls(self.prefix, detail=False)}

    def read(self, name):
        return self.fs.cat_file(self.path(name))

    def put(self, name, data):
        path = self.path(name)
        if self.fs.exists(path):
            require(
                digest(self.read(name)) == digest(data), f"immutable storage object differs: {name}"
            )
            return
        self.fs.pipe_file(path, data)
        self.fs.invalidate_cache(self.prefix)
        require(digest(self.read(name)) == digest(data), f"external read-back failed: {name}")

    def record(self, name, document):
        self.put(name, json_bytes(document))

    def commit(self, label, root, paths, metadata):
        root = Path(root).resolve()
        files = {}
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=3) as archive:
            for item in sorted({Path(p).resolve() for p in paths}):
                require(item.is_file() and item.is_relative_to(root), "archive member outside root")
                name = item.relative_to(root).as_posix()
                value = item.read_bytes()
                files[name] = {"sha256": digest(value), "bytes": len(value)}
                archive.writestr(ZipInfo(name), value, compress_type=ZIP_DEFLATED, compresslevel=3)
        data = buffer.getvalue()
        archive_hash = digest(data)
        name = f"{label}-{archive_hash}.zip"
        self.put(name, data)
        receipt = {
            "plan_sha256": self.plan_hash,
            "label": label,
            "archive": name,
            "archive_sha256": archive_hash,
            "archive_bytes": len(data),
            "files": files,
            "metadata": metadata,
        }
        self.record(f"{label}.receipt.json", receipt)
        return receipt

    def receipt(self, label):
        receipt = json.loads(self.read(f"{label}.receipt.json"))
        require(
            receipt["plan_sha256"] == self.plan_hash and receipt["label"] == label,
            "receipt is from a different sweep plan",
        )
        return receipt

    def verify(self, label, *, restore_root=None, json_suffix=None):
        receipt = self.receipt(label)
        data = self.read(receipt["archive"])
        require(
            len(data) == receipt["archive_bytes"] and digest(data) == receipt["archive_sha256"],
            f"archive hash mismatch: {label}",
        )
        documents = {}
        with ZipFile(io.BytesIO(data)) as archive:
            require(
                sorted(archive.namelist()) == sorted(receipt["files"]),
                "archive member inventory differs",
            )
            for name, spec in receipt["files"].items():
                member = PurePosixPath(name)
                require(
                    not member.is_absolute() and ".." not in member.parts, "unsafe archive member"
                )
                value = archive.read(name)
                require(
                    len(value) == spec["bytes"] and digest(value) == spec["sha256"],
                    f"archive member hash mismatch: {name}",
                )
                if restore_root is not None:
                    root = Path(restore_root).resolve()
                    target = (root / member).resolve()
                    require(target.is_relative_to(root), "archive restore escapes destination")
                    atomic_file(target, lambda tmp, data=value: tmp.write_bytes(data))
                if json_suffix and name.endswith(json_suffix):
                    documents[name] = json.loads(value)
        return receipt, documents
