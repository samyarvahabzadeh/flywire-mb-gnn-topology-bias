"""Package frozen differential-conditioning inputs for the Molab CUDA runner."""

import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dist/differential-molab.zip"
PLAN = ROOT / "configs/exploratory_similarity_gradient.json"
RUNNER = ROOT / "scripts/molab_similarity_gradient.py"
OUTPUT = ROOT / "dist/differential-similarity-gradient.zip"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


with ZipFile(SOURCE) as old:
    source_provenance = json.loads(old.read("bundle.json"))
    files = {
        "projection.npz": old.read("projection.npz"),
        "masks.npz": old.read("masks.npz"),
        "plan.json": PLAN.read_bytes(),
        "run.py": RUNNER.read_bytes(),
    }
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
bundle = {
    "files": {name: digest(data) for name, data in files.items()},
    "git_commit": commit,
    "source_bundle_sha256": digest(SOURCE.read_bytes()),
    "source_graph_manifest_sha256": source_provenance["real_graph_manifest_sha256"],
    "input_mask_catalog_sha256": source_provenance["source_masks_sha256"],
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as packed:
    for name, data in files.items():
        packed.writestr(name, data)
    packed.writestr("bundle.json", json.dumps(bundle, sort_keys=True, indent=2))
print(OUTPUT, digest(OUTPUT.read_bytes()), OUTPUT.stat().st_size, commit)
