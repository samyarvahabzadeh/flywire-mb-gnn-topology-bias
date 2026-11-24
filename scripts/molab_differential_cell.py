def run_differential_on_molab(smoke=True):
    import hashlib
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

    from gdrive_fsspec import GoogleDriveFileSystem

    started = time.monotonic()
    drive = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    parent = "root/FlyWireMB-v12-sweep"
    folder = Path("/marimo/differential-04bc07a")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "differential-molab.zip"
    drive.get_file(f"{parent}/{archive.name}", str(archive))
    expected = "0d2562989d88adfafaddbe3bae1b3289fd861dd19696e0a0f9040b18937bb4aa"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Differential conditioning payload checksum differs")
    with ZipFile(archive) as payload:
        payload.extractall(folder)
    output = folder / ("smoke" if smoke else "results")
    remote = f"{parent}/differential-04bc07a/{output.name}"
    drive.makedirs(remote, exist_ok=True)
    output.mkdir(exist_ok=True)
    for receipt in drive.glob(f"{remote}/seed-*.json"):
        name = Path(receipt).stem
        for ext in (".parquet", ".json"):
            drive.get_file(f"{remote}/{name}{ext}", str(output / (name + ext)))
    command = [sys.executable, "-u", "run.py", "--output", str(output)]
    if smoke:
        command.append("--smoke")

    def archive_file(path):
        target = f"{remote}/{path.name}"
        data = path.read_bytes()
        if not drive.exists(target):
            drive.pipe_file(target, data)
        if drive.cat_file(target) != data:
            raise ValueError(f"Drive readback differs for {path.name}")

    with subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE, text=True) as job:
        for line in job.stdout:
            print(line.strip(), flush=True)
            if line.startswith("VERIFIED_SEED "):
                seed = int(line.split()[1])
                for ext in (".parquet", ".json"):
                    archive_file(output / f"seed-{seed}{ext}")
        if job.wait():
            raise RuntimeError("Differential conditioning worker failed")
    for name in ("seed-contrasts.parquet", "MANIFEST.json"):
        archive_file(output / name)
    print("DRIVE_VERIFIED", remote, f"{time.monotonic() - started:.1f}s", flush=True)
    if not smoke:
        manifest = json.loads((output / "MANIFEST.json").read_text())
        names = ["MANIFEST.json", "seed-contrasts.parquet"]
        for shard in manifest["shards"]:
            names.extend((shard, str(Path(shard).with_suffix(".json"))))
        result = output / "differential-molab-results.zip"
        with ZipFile(result, "w", compression=ZIP_DEFLATED) as packed:
            for name in sorted(names):
                payload = (output / name).read_bytes()
                if drive.cat_file(f"{remote}/{name}") != payload:
                    raise ValueError(f"Final readback differs for {name}")
                packed.writestr(ZipInfo(name), payload, compress_type=ZIP_DEFLATED)
        archive_file(result)
        print(
            "ARCHIVE_VERIFIED",
            hashlib.sha256(result.read_bytes()).hexdigest(),
            result.stat().st_size,
            flush=True,
        )


run_differential_on_molab(smoke=True)
