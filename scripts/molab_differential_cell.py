def run_differential_on_molab(smoke=True):
    import hashlib
    import subprocess
    import sys
    from pathlib import Path
    from zipfile import ZipFile

    from gdrive_fsspec import GoogleDriveFileSystem

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
    print("DRIVE_VERIFIED", remote)


run_differential_on_molab(smoke=True)
