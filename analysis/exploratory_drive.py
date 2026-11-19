"""Run the exploratory analysis with bounded prefetch of frozen Drive archives."""

import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from threading import local

from confirmatory_data import require
from exploratory_mechanism import publish, run
from gdrive_fsspec import GoogleDriveFileSystem

BATCH_ARCHIVE = re.compile(r"batch-(\d{4})(?:\.receipt\.json|-[a-f0-9]{64}\.zip)")


def drive():
    return GoogleDriveFileSystem(
        skip_instance_cache=True, auth_kwargs={"use_local_webserver": False}
    )


class ArchiveReader:
    def __init__(self, fs, prefix, executor):
        self.entries = {}
        self.batches = defaultdict(list)
        for entry in fs.ls(prefix, detail=True):
            name = PurePosixPath(entry["name"]).name
            require(name not in self.entries, f"duplicate Drive item in analysis archive: {name}")
            self.entries[name] = entry["id"]
            if match := BATCH_ARCHIVE.fullmatch(name):
                self.batches[int(match[1])].append(name)
        require(set(self.batches) == set(range(305)), "Drive batch inventory changed")
        for index, names in self.batches.items():
            receipt = f"batch-{index:04d}.receipt.json"
            require(receipt in names and len(names) >= 2, f"incomplete Drive batch {index}")
            if len(names) > 2:
                # Only the verified receipt can select an archive when retries left orphan files.
                self.batches[index] = [receipt]
        self.executor = executor
        self.thread = local()
        self.pending = {}
        self.consumed = set()

    def download(self, name):
        if not hasattr(self.thread, "fs"):
            self.thread.fs = drive()
        request = self.thread.fs.files.get_media(fileId=self.entries[name], supportsAllDrives=True)
        return request.execute(num_retries=4)

    def __call__(self, name):
        if match := BATCH_ARCHIVE.fullmatch(name):
            start = int(match[1])
            for index in range(start, min(start + 12, len(self.batches))):
                for item in self.batches[index]:
                    if item not in self.pending and item not in self.consumed:
                        self.pending[item] = self.executor.submit(self.download, item)
        if name not in self.pending:
            self.pending[name] = self.executor.submit(self.download, name)
        result = self.pending.pop(name).result()
        self.consumed.add(name)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-prefix", required=True)
    parser.add_argument("--export-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.export_prefix.rstrip("/") != args.drive_prefix.rstrip("/"),
        "choose a separate Drive export prefix",
    )
    fs = drive()
    if not (args.output / "MANIFEST.json").exists():
        with ThreadPoolExecutor(max_workers=8) as executor:
            reader = ArchiveReader(fs, args.drive_prefix, executor)
            print(f"EXPLORATORY_DRIVE_INVENTORY {len(reader.entries)}", flush=True)
            run(reader, args.output, workers=4)
    publish(fs, args.export_prefix, args.output)


if __name__ == "__main__":
    main()
